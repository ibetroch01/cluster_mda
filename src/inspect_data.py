"""Inspect raw AWV cycling count CSV schemas.

This script does not perform feature engineering. It inventories raw CSV files,
normalizes detected column names, checks count date ranges and intervals, and
writes a compact JSON schema summary for downstream pipeline design.
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
LEGACY_RAW_DIR = PROJECT_ROOT / "data" / "awv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "data_schema_summary.json"

# Raw AWV files do not always arrive with perfectly consistent headers, so the
# exploration step detects schemas instead of assuming them.
COUNT_FILE_PATTERN = re.compile(r"^data-\d{4}-\d{2}\.csv$", re.IGNORECASE)
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "counts": ["site_id", "richting", "type", "van", "tot", "aantal"],
    "sites": ["site_id", "site_nr", "long", "lat", "naam", "domein", "wegnr", "district", "gemeente", "interval", "datum_van"],
    "directions": ["site_id", "richting", "naam"],
}
HEADER_MATCH_MINIMUM: dict[str, int] = {"counts": 3, "sites": 4, "directions": 2}


class DataInspectionError(RuntimeError):
    """Raised when the raw data folder cannot be inspected."""


@dataclass(frozen=True)
class CsvSchema:
    """Detected schema metadata for one CSV file."""

    path: str
    file_type: str
    delimiter: str
    header_present: bool
    columns: list[str]
    raw_first_row: list[str]
    notes: list[str] = field(default_factory=list)


def normalize_column_name(value: object, index: int | None = None) -> str:
    """Normalize a source column name into a stable snake_case identifier."""
    # A single naming convention makes the later scripts independent of small
    # CSV header differences such as accents, spaces or capital letters.
    if pd.isna(value):
        text = ""
    else:
        text = str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text in {"siteid", "site_id_"}:
        text = "site_id"
    if not text:
        text = f"column_{index}" if index is not None else "column"
    return text


def deduplicate_columns(columns: Iterable[str]) -> list[str]:
    """Make normalized column names unique while preserving order."""
    counts: dict[str, int] = {}
    unique: list[str] = []
    for column in columns:
        base = column
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            unique.append(base)
        else:
            unique.append(f"{base}_{counts[base]}")
    return unique


def classify_file(path: Path) -> str:
    """Classify a raw CSV by filename."""
    name = path.name.lower()
    if name == "sites.csv":
        return "sites"
    if name in {"richtingen.csv", "directions.csv"}:
        return "directions"
    if COUNT_FILE_PATTERN.match(name):
        return "counts"
    return "unknown"


def detect_delimiter(path: Path) -> str:
    """Detect a CSV delimiter with a conservative comma fallback."""
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


def read_preview(path: Path, delimiter: str, nrows: int = 5) -> pd.DataFrame:
    """Read a small headerless preview used for schema detection."""
    return pd.read_csv(
        path,
        sep=delimiter,
        header=None,
        nrows=nrows,
        dtype="string",
        encoding="utf-8-sig",
        on_bad_lines="warn",
    )


def canonical_columns(file_type: str, n_columns: int) -> tuple[list[str], list[str]]:
    """Return canonical columns for known headerless files, plus notes."""
    expected = EXPECTED_COLUMNS.get(file_type)
    if not expected:
        return [f"column_{idx}" for idx in range(1, n_columns + 1)], ["unknown file type; generic columns assigned"]
    if n_columns == len(expected):
        return expected.copy(), ["headerless file detected; canonical columns assigned by file type and column count"]
    if n_columns > len(expected):
        extra = [f"extra_column_{idx}" for idx in range(1, n_columns - len(expected) + 1)]
        return [*expected, *extra], ["headerless file detected; expected columns plus generic extras assigned"]
    return [f"column_{idx}" for idx in range(1, n_columns + 1)], [
        f"column count {n_columns} does not match expected {len(expected)} for {file_type}; generic columns assigned"
    ]


def detect_schema(path: Path, raw_dir: Path) -> CsvSchema:
    """Detect normalized columns, delimiter, and header status for one CSV."""
    file_type = classify_file(path)
    delimiter = detect_delimiter(path)
    preview = read_preview(path, delimiter)
    if preview.empty:
        raise DataInspectionError(f"{path} is empty")

    first_row = ["" if pd.isna(value) else str(value) for value in preview.iloc[0].tolist()]
    normalized_first_row = deduplicate_columns(
        normalize_column_name(value, idx) for idx, value in enumerate(first_row, start=1)
    )
    expected = set(EXPECTED_COLUMNS.get(file_type, []))
    n_matches = len(set(normalized_first_row) & expected)
    # Some AWV files are headerless; a row is treated as a header only when it
    # matches enough expected columns for that file type.
    header_present = bool(expected and n_matches >= HEADER_MATCH_MINIMUM.get(file_type, 2))

    notes: list[str] = []
    if header_present:
        columns = normalized_first_row
        notes.append("header row detected from expected AWV column names")
    else:
        columns, notes = canonical_columns(file_type, len(first_row))

    return CsvSchema(
        path=str(path.relative_to(PROJECT_ROOT)),
        file_type=file_type,
        delimiter=delimiter,
        header_present=header_present,
        columns=columns,
        raw_first_row=first_row,
        notes=notes,
    )


def read_csv_with_schema(path: Path, schema: CsvSchema, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV using the detected schema and normalized columns."""
    if schema.header_present:
        data = pd.read_csv(path, sep=schema.delimiter, dtype="string", encoding="utf-8-sig", on_bad_lines="warn")
        data.columns = deduplicate_columns(normalize_column_name(column, idx) for idx, column in enumerate(data.columns, start=1))
        if columns is not None:
            missing = sorted(set(columns) - set(data.columns))
            if missing:
                raise DataInspectionError(f"{schema.path} missing columns: {', '.join(missing)}")
            return data[columns]
        return data

    read_kwargs: dict[str, object] = {
        "sep": schema.delimiter,
        "header": None,
        "names": schema.columns,
        "dtype": "string",
        "encoding": "utf-8-sig",
        "on_bad_lines": "warn",
    }
    if columns is not None:
        read_kwargs["usecols"] = columns
    return pd.read_csv(path, **read_kwargs)


def unique_strings(series: pd.Series) -> list[str]:
    """Return sorted unique non-empty string values."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    return sorted(values.unique().tolist())


def inspect_count_file(path: Path, schema: CsvSchema) -> dict[str, object]:
    """Inspect one monthly count CSV for dates, count types, and intervals."""
    required = ["type", "van", "tot"]
    result: dict[str, object] = {
        "path": schema.path,
        "columns": schema.columns,
        "date_min": None,
        "date_max": None,
        "type_values": [],
        "interval_minutes_unique": [],
        "interval_is_15_minutes": False,
        "n_rows_checked": 0,
        "errors": [],
    }
    try:
        data = read_csv_with_schema(path, schema, columns=required)
        result["n_rows_checked"] = int(len(data))
        van = pd.to_datetime(data["van"], errors="coerce")
        tot = pd.to_datetime(data["tot"], errors="coerce")
        intervals = ((tot - van).dt.total_seconds() / 60.0).dropna()
        interval_values = sorted({float(value) for value in intervals.unique().tolist()})
        result.update(
            {
                "date_min": van.min().isoformat() if van.notna().any() else None,
                "date_max": tot.max().isoformat() if tot.notna().any() else None,
                "type_values": unique_strings(data["type"]),
                "interval_minutes_unique": interval_values,
                "interval_is_15_minutes": interval_values == [15.0],
            }
        )
    except Exception as exc:  # noqa: BLE001 - keep per-file inspection resilient
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def inspect_sites_file(path: Path, schema: CsvSchema) -> dict[str, object]:
    """Inspect sites metadata, especially interval values."""
    result: dict[str, object] = {
        "path": schema.path,
        "columns": schema.columns,
        "interval_values": [],
        "interval_is_15_minutes": None,
        "errors": [],
    }
    try:
        data = read_csv_with_schema(path, schema)
        if "interval" not in data.columns:
            result["errors"].append("interval column not detected")
            return result
        intervals = pd.to_numeric(data["interval"], errors="coerce").dropna()
        values = sorted({float(value) for value in intervals.unique().tolist()})
        result["interval_values"] = values
        result["interval_is_15_minutes"] = values == [15.0]
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def inspect_directions_file(path: Path, schema: CsvSchema) -> dict[str, object]:
    """Inspect directions metadata."""
    result: dict[str, object] = {"path": schema.path, "columns": schema.columns, "richting_values": [], "errors": []}
    try:
        data = read_csv_with_schema(path, schema)
        if "richting" in data.columns:
            result["richting_values"] = unique_strings(data["richting"])
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def resolve_raw_dir(raw_dir: Path) -> Path:
    """Use data/raw by default, with a legacy data/awv fallback for older layouts."""
    if raw_dir.exists() and any(raw_dir.glob("*.csv")):
        return raw_dir
    if raw_dir == DEFAULT_RAW_DIR and LEGACY_RAW_DIR.exists() and any(LEGACY_RAW_DIR.glob("*.csv")):
        return LEGACY_RAW_DIR
    return raw_dir


def build_summary(raw_dir: Path) -> dict[str, object]:
    """Build the full data schema summary."""
    raw_dir = resolve_raw_dir(raw_dir)
    if not raw_dir.exists():
        raise DataInspectionError(f"Raw data directory does not exist: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise DataInspectionError(f"No CSV files found in {raw_dir}")

    schemas: list[CsvSchema] = []
    errors: list[str] = []
    for path in csv_files:
        try:
            schemas.append(detect_schema(path, raw_dir))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.relative_to(PROJECT_ROOT)}: {type(exc).__name__}: {exc}")

    schema_by_path = {schema.path: schema for schema in schemas}
    schemas_by_file_type: dict[str, list[list[str]]] = {}
    for schema in schemas:
        schemas_by_file_type.setdefault(schema.file_type, [])
        if schema.columns not in schemas_by_file_type[schema.file_type]:
            schemas_by_file_type[schema.file_type].append(schema.columns)

    count_results: list[dict[str, object]] = []
    site_result: dict[str, object] | None = None
    direction_result: dict[str, object] | None = None
    for path in csv_files:
        rel = str(path.relative_to(PROJECT_ROOT))
        schema = schema_by_path.get(rel)
        if schema is None:
            continue
        if schema.file_type == "counts":
            count_results.append(inspect_count_file(path, schema))
        elif schema.file_type == "sites":
            site_result = inspect_sites_file(path, schema)
        elif schema.file_type == "directions":
            direction_result = inspect_directions_file(path, schema)

    all_date_mins = [pd.Timestamp(item["date_min"]) for item in count_results if item.get("date_min")]
    all_date_maxs = [pd.Timestamp(item["date_max"]) for item in count_results if item.get("date_max")]
    all_types = sorted({value for item in count_results for value in item.get("type_values", [])})
    all_intervals = sorted({value for item in count_results for value in item.get("interval_minutes_unique", [])})
    count_errors = [error for item in count_results for error in item.get("errors", [])]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
        "n_csv_files": len(csv_files),
        "csv_files": [asdict(schema) for schema in schemas],
        "schemas_by_file_type": schemas_by_file_type,
        "counts": {
            "n_files": len(count_results),
            "date_min": min(all_date_mins).isoformat() if all_date_mins else None,
            "date_max": max(all_date_maxs).isoformat() if all_date_maxs else None,
            "type_values": all_types,
            "interval_minutes_unique": all_intervals,
            "interval_is_15_minutes": all_intervals == [15.0] and not count_errors,
            "files": count_results,
        },
        "sites": site_result,
        "directions": direction_result,
        "errors": errors,
    }


def print_summary(summary: dict[str, object]) -> None:
    """Print a human-readable inspection summary."""
    print(f"Raw data directory: {summary['raw_dir']}")
    print(f"CSV files found: {summary['n_csv_files']}")
    for item in summary["csv_files"]:
        print(f"- {item['path']} [{item['file_type']}]")

    print("\nDetected columns by file type:")
    for file_type, schemas in summary["schemas_by_file_type"].items():
        print(f"- {file_type}:")
        for columns in schemas:
            print(f"  {columns}")

    counts = summary["counts"]
    print("\nCount files:")
    print(f"- files: {counts['n_files']}")
    print(f"- overall date range: {counts['date_min']} -> {counts['date_max']}")
    print(f"- unique type values: {counts['type_values']}")
    print(f"- unique intervals in minutes: {counts['interval_minutes_unique']}")
    print(f"- all count intervals are 15 minutes: {counts['interval_is_15_minutes']}")
    print("- per-file date ranges:")
    for item in counts["files"]:
        print(f"  {item['path']}: {item['date_min']} -> {item['date_max']}")

    sites = summary.get("sites")
    if sites:
        print("\nSites metadata:")
        print(f"- columns: {sites['columns']}")
        print(f"- interval values: {sites['interval_values']}")
        print(f"- site interval is 15 minutes: {sites['interval_is_15_minutes']}")

    directions = summary.get("directions")
    if directions:
        print("\nDirections metadata:")
        print(f"- columns: {directions['columns']}")
        print(f"- richting values: {directions['richting_values']}")

    if summary["errors"]:
        print("\nInspection errors:")
        for error in summary["errors"]:
            print(f"- {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw AWV cycling count CSV files.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing raw AWV CSV files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON summary output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_summary(args.raw_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print_summary(summary)
        print(f"\nWrote machine-readable summary: {args.output.relative_to(PROJECT_ROOT)}")
        return 0
    except DataInspectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
