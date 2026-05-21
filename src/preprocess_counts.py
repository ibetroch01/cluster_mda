"""Preprocess raw AWV cyclist counts to site-level interval time series.

This script creates an auditable site-level time series for 2023-2025. It keeps
valid 15-minute intervals and valid 75-minute Europe/Brussels spring DST rows as
single measurement intervals using their start timestamp. It does not split,
divide, or impute intervals.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diagnose_intervals import brussels_spring_dst_date, last_sunday  # noqa: E402
from src.inspect_data import (  # noqa: E402
    DataInspectionError,
    DEFAULT_RAW_DIR,
    PROJECT_ROOT,
    classify_file,
    detect_schema,
    read_csv_with_schema,
    resolve_raw_dir,
)

START_DATE = pd.Timestamp("2023-01-01")
END_DATE = pd.Timestamp("2026-01-01")
VALID_INTERVAL_STATUSES = {"normal_15min", "dst_spring_75min"}
VALID_MINUTES = {0, 15, 30, 45}
BRUSSELS_TZ = ZoneInfo("Europe/Brussels")
MONTHLY_COUNT_PATTERN = re.compile(r"^data-(\d{4})-(\d{2})\.csv$", re.IGNORECASE)
REQUIRED_COUNT_COLUMNS = ["site_id", "richting", "type", "van", "tot", "aantal"]

DEFAULT_PROCESSED_OUTPUT = PROJECT_ROOT / "data" / "processed" / "counts_site_level_2023_2025.parquet"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "preprocessing_report.json"
DEFAULT_ANOMALOUS_OUTPUT = PROJECT_ROOT / "outputs" / "anomalous_duration_rows.csv"
DEFAULT_NEGATIVE_OUTPUT = PROJECT_ROOT / "outputs" / "negative_count_rows.csv"
DEFAULT_DUPLICATE_OUTPUT = PROJECT_ROOT / "outputs" / "duplicate_interval_start_report.csv"
DEFAULT_DST_DUPLICATE_OUTPUT = PROJECT_ROOT / "outputs" / "dst_duplicate_diagnostics.json"

WARNING_COLUMNS = [
    "source_file",
    "site_id",
    "richting",
    "type",
    "van",
    "tot",
    "aantal",
    "duration_minutes",
    "interval_status",
    "date",
    "year",
    "month",
    "day_of_week",
    "hour",
    "minute",
    "quarter_index",
]
DUPLICATE_COLUMNS = [
    "duplicate_type",
    "dst_duplicate_status",
    "site_id",
    "richting",
    "van",
    "tot",
    "date",
    "year",
    "month",
    "hour",
    "minute",
    "n_rows",
    "n_source_files",
    "source_files",
    "aantal_sum",
    "aantal_values",
]
PROCESSED_COLUMNS = [
    "site_id",
    "interval_start",
    "count",
    "date",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "hour",
    "minute",
    "quarter_index",
    "interval_status",
    "has_dst_spring_75min",
    "n_direction_rows",
    "n_directions",
    "duration_minutes_min",
    "duration_minutes_max",
]


def monthly_count_files(raw_dir: Path) -> list[Path]:
    """Return AWV monthly count CSV files for the 2023-2025 processing period."""
    selected: list[Path] = []
    for path in sorted(raw_dir.glob("data-*.csv")):
        match = MONTHLY_COUNT_PATTERN.match(path.name)
        if not match:
            continue
        year = int(match.group(1))
        if 2023 <= year <= 2025:
            selected.append(path)
    return selected


def brussels_autumn_dst_date(year: int) -> pd.Timestamp:
    """Return the Europe/Brussels autumn DST transition date for a year."""
    transition_date = last_sunday(year, 10)
    before = pd.Timestamp(year=year, month=10, day=transition_date.day, hour=1, tz=BRUSSELS_TZ).utcoffset()
    after = pd.Timestamp(year=year, month=10, day=transition_date.day, hour=4, tz=BRUSSELS_TZ).utcoffset()
    if before is None or after is None or after >= before:
        raise DataInspectionError(f"Could not validate Brussels autumn DST transition for {year}")
    return transition_date


def is_autumn_dst_repeated_hour(van_dt: pd.Series) -> pd.Series:
    """Return True for local timestamps in the repeated autumn DST hour."""
    years = sorted(van_dt.dt.year.dropna().unique())
    transition_dates = {year: brussels_autumn_dst_date(int(year)) for year in years}
    expected_dates = van_dt.dt.year.map(transition_dates)
    return van_dt.dt.normalize().eq(expected_dates) & van_dt.dt.hour.eq(2)


def is_spring_dst_75(van_dt: pd.Series, duration_minutes: pd.Series) -> pd.Series:
    """Return True for 75-minute rows on Brussels spring DST transition dates."""
    years = sorted(van_dt.dt.year.dropna().unique())
    transition_dates = {year: brussels_spring_dst_date(int(year)) for year in years}
    expected_dates = van_dt.dt.year.map(transition_dates)
    return duration_minutes.eq(75.0) & van_dt.dt.normalize().eq(expected_dates)


def load_metadata(raw_dir: Path) -> dict[str, object]:
    """Load sites and directions metadata for audit counts only."""
    result: dict[str, object] = {
        "sites_path": None,
        "directions_path": None,
        "n_sites": None,
        "n_direction_rows": None,
        "site_columns": [],
        "direction_columns": [],
        "errors": [],
    }
    sites_path = raw_dir / "sites.csv"
    directions_path = raw_dir / "richtingen.csv"
    if sites_path.exists():
        try:
            schema = detect_schema(sites_path, raw_dir)
            sites = read_csv_with_schema(sites_path, schema)
            result.update(
                {
                    "sites_path": str(sites_path.relative_to(PROJECT_ROOT)),
                    "n_sites": int(sites["site_id"].nunique()) if "site_id" in sites.columns else int(len(sites)),
                    "site_columns": list(sites.columns),
                }
            )
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"sites.csv: {type(exc).__name__}: {exc}")
    else:
        result["errors"].append("sites.csv not found")

    if directions_path.exists():
        try:
            schema = detect_schema(directions_path, raw_dir)
            directions = read_csv_with_schema(directions_path, schema)
            result.update(
                {
                    "directions_path": str(directions_path.relative_to(PROJECT_ROOT)),
                    "n_direction_rows": int(len(directions)),
                    "direction_columns": list(directions.columns),
                }
            )
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"richtingen.csv: {type(exc).__name__}: {exc}")
    else:
        result["errors"].append("richtingen.csv not found")
    return result


def load_count_file(path: Path) -> pd.DataFrame:
    """Load and normalize one monthly count CSV."""
    if classify_file(path) != "counts":
        raise DataInspectionError(f"Not a monthly count CSV: {path}")
    schema = detect_schema(path, path.parent)
    data = read_csv_with_schema(path, schema, columns=REQUIRED_COUNT_COLUMNS)
    missing = sorted(set(REQUIRED_COUNT_COLUMNS) - set(data.columns))
    if missing:
        raise DataInspectionError(f"{path.name} missing required columns after normalization: {', '.join(missing)}")
    data = data.copy()
    data["source_file"] = str(path.relative_to(PROJECT_ROOT))
    return data


def normalize_and_filter_counts(data: pd.DataFrame) -> pd.DataFrame:
    """Parse fields and keep requested FIETSERS rows in the selected date window."""
    data = data.copy()
    data["van_dt"] = pd.to_datetime(data["van"], errors="coerce")
    data["tot_dt"] = pd.to_datetime(data["tot"], errors="coerce")
    data["site_id"] = pd.to_numeric(data["site_id"], errors="coerce").astype("Int64")
    data["richting"] = data["richting"].astype("string").str.strip()
    data["aantal"] = pd.to_numeric(data["aantal"], errors="coerce")
    type_text = data["type"].astype("string").str.strip().str.casefold()
    mask = data["van_dt"].ge(START_DATE) & data["van_dt"].lt(END_DATE) & type_text.eq("fietsers")
    data = data.loc[mask].copy()
    data["duration_minutes"] = (data["tot_dt"] - data["van_dt"]).dt.total_seconds() / 60.0
    data["interval_status"] = "anomalous_duration"
    data.loc[data["duration_minutes"].eq(15.0), "interval_status"] = "normal_15min"
    data.loc[data["duration_minutes"].eq(75.0), "interval_status"] = "dst_spring_75min"
    return data


def add_time_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add interval_start and calendar columns from van."""
    data = data.copy()
    data["interval_start"] = data["van_dt"]
    data["date"] = data["interval_start"].dt.date.astype("string")
    data["year"] = data["interval_start"].dt.year.astype("Int64")
    data["month"] = data["interval_start"].dt.month.astype("Int64")
    data["day_of_week"] = data["interval_start"].dt.dayofweek.astype("Int64")
    data["is_weekend"] = data["day_of_week"].isin([5, 6])
    data["hour"] = data["interval_start"].dt.hour.astype("Int64")
    data["minute"] = data["interval_start"].dt.minute.astype("Int64")
    data["quarter_index"] = (data["hour"] * 4 + data["minute"] // 15).astype("Int64")
    return data


def warning_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return a compact warning dataframe with stable columns."""
    frame = data.copy()
    for column in WARNING_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[WARNING_COLUMNS].reset_index(drop=True)


def aggregate_duplicate_subset(data: pd.DataFrame, key_columns: list[str], duplicate_type: str) -> pd.DataFrame:
    """Build duplicate diagnostics from rows that duplicate the requested key."""
    duplicate_mask = data.duplicated(key_columns, keep=False)
    if not bool(duplicate_mask.any()):
        return pd.DataFrame(columns=DUPLICATE_COLUMNS)

    source = data.loc[duplicate_mask].copy()
    records = []
    for key_values, group in source.groupby(key_columns, sort=True, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        record = dict(zip(key_columns, key_values))
        van_dt = pd.to_datetime(group["van_dt"], errors="coerce")
        dst_mask = is_autumn_dst_repeated_hour(van_dt) if van_dt.notna().any() else pd.Series(False, index=group.index)
        dst_status = "likely_dst_repeated_hour" if bool(dst_mask.all()) and len(group) > 1 else "suspicious_duplicate"
        record.update(
            {
                "duplicate_type": duplicate_type,
                "dst_duplicate_status": dst_status,
                "date": van_dt.dt.date.astype("string").iloc[0] if van_dt.notna().any() else pd.NA,
                "year": int(van_dt.dt.year.iloc[0]) if van_dt.notna().any() else pd.NA,
                "month": int(van_dt.dt.month.iloc[0]) if van_dt.notna().any() else pd.NA,
                "hour": int(van_dt.dt.hour.iloc[0]) if van_dt.notna().any() else pd.NA,
                "minute": int(van_dt.dt.minute.iloc[0]) if van_dt.notna().any() else pd.NA,
                "n_rows": int(len(group)),
                "n_source_files": int(group["source_file"].nunique()),
                "source_files": "|".join(sorted(group["source_file"].astype(str).unique())),
                "aantal_sum": float(group["aantal"].sum(skipna=True)),
                "aantal_values": "|".join(group["aantal"].astype("string").fillna("<NA>").tolist()),
            }
        )
        records.append(record)
    output = pd.DataFrame.from_records(records)
    for column in DUPLICATE_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    return output[DUPLICATE_COLUMNS]


def build_duplicate_diagnostics(valid_rows: pd.DataFrame) -> pd.DataFrame:
    """Report exact and local interval-start duplicates before direction aggregation."""
    exact = aggregate_duplicate_subset(
        valid_rows,
        key_columns=["site_id", "richting", "van", "tot"],
        duplicate_type="exact_site_direction_van_tot",
    )
    local = aggregate_duplicate_subset(
        valid_rows,
        key_columns=["site_id", "richting", "van"],
        duplicate_type="local_site_direction_van",
    )
    return pd.concat([exact, local], ignore_index=True).sort_values(
        ["date", "site_id", "richting", "duplicate_type"], na_position="last"
    ).reset_index(drop=True)


def aggregate_site_level(valid_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over directions at original site_id and interval_start."""
    rows = valid_rows.copy()
    rows["has_dst_spring_75min"] = rows["interval_status"].eq("dst_spring_75min")
    grouped = rows.groupby(["site_id", "interval_start"], as_index=False, observed=True, dropna=False)
    output = grouped.agg(
        count=("aantal", "sum"),
        date=("date", "first"),
        year=("year", "first"),
        month=("month", "first"),
        day_of_week=("day_of_week", "first"),
        is_weekend=("is_weekend", "first"),
        hour=("hour", "first"),
        minute=("minute", "first"),
        quarter_index=("quarter_index", "first"),
        has_dst_spring_75min=("has_dst_spring_75min", "max"),
        n_direction_rows=("richting", "size"),
        n_directions=("richting", "nunique"),
        duration_minutes_min=("duration_minutes", "min"),
        duration_minutes_max=("duration_minutes", "max"),
    )
    output["interval_status"] = "normal_15min"
    output.loc[output["has_dst_spring_75min"], "interval_status"] = "dst_spring_75min"
    output = output[PROCESSED_COLUMNS].sort_values(["site_id", "interval_start"]).reset_index(drop=True)
    return output


def build_report(
    raw_dir: Path,
    files: list[Path],
    metadata: dict[str, object],
    loaded_rows: int,
    filtered_rows: int,
    anomalous_rows: pd.DataFrame,
    negative_rows: pd.DataFrame,
    invalid_minute_rows: int,
    duplicate_report: pd.DataFrame,
    rows_kept_before_aggregation: int,
    processed: pd.DataFrame,
    duration_counts_kept: dict[str, int],
    interval_status_counts_kept: dict[str, int],
    minute_values_kept: set[int],
    file_errors: list[str],
) -> dict[str, object]:
    """Create machine-readable preprocessing report."""
    duplicate_counts = duplicate_report["dst_duplicate_status"].value_counts().to_dict() if not duplicate_report.empty else {}
    exact_duplicates = duplicate_report.loc[duplicate_report["duplicate_type"].eq("exact_site_direction_van_tot")]
    local_duplicates = duplicate_report.loc[duplicate_report["duplicate_type"].eq("local_site_direction_van")]
    spring_dates = {str(year): brussels_spring_dst_date(year).date().isoformat() for year in [2023, 2024, 2025]}
    autumn_dates = {str(year): brussels_autumn_dst_date(year).date().isoformat() for year in [2023, 2024, 2025]}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
        "date_filter": {"start_inclusive": START_DATE.isoformat(), "end_exclusive": END_DATE.isoformat()},
        "type_filter": "FIETSERS case-insensitive after stripping whitespace",
        "monthly_files_loaded": len(files),
        "monthly_files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "metadata": metadata,
        "loaded_raw_rows": int(loaded_rows),
        "rows_after_date_and_type_filter": int(filtered_rows),
        "anomalous_duration_rows": int(len(anomalous_rows)),
        "negative_count_rows": int(len(negative_rows)),
        "invalid_minute_rows": int(invalid_minute_rows),
        "rows_kept_before_direction_aggregation": int(rows_kept_before_aggregation),
        "processed_site_interval_rows": int(len(processed)),
        "n_sites_processed": int(processed["site_id"].nunique()) if not processed.empty else 0,
        "duration_minutes_counts_kept": {str(key): int(value) for key, value in duration_counts_kept.items()},
        "interval_status_counts_kept": {str(key): int(value) for key, value in interval_status_counts_kept.items()},
        "minute_values_kept": sorted(int(value) for value in minute_values_kept),
        "spring_dst_transition_dates_europe_brussels": spring_dates,
        "autumn_dst_transition_dates_europe_brussels": autumn_dates,
        "duplicate_diagnostics": {
            "duplicate_groups_total": int(len(duplicate_report)),
            "exact_site_direction_van_tot_groups": int(len(exact_duplicates)),
            "local_site_direction_van_groups": int(len(local_duplicates)),
            "by_dst_duplicate_status": {str(key): int(value) for key, value in duplicate_counts.items()},
        },
        "outputs": {
            "processed_parquet": str(DEFAULT_PROCESSED_OUTPUT.relative_to(PROJECT_ROOT)),
            "preprocessing_report": str(DEFAULT_REPORT_OUTPUT.relative_to(PROJECT_ROOT)),
            "anomalous_duration_rows": str(DEFAULT_ANOMALOUS_OUTPUT.relative_to(PROJECT_ROOT)),
            "negative_count_rows": str(DEFAULT_NEGATIVE_OUTPUT.relative_to(PROJECT_ROOT)),
            "duplicate_interval_start_report": str(DEFAULT_DUPLICATE_OUTPUT.relative_to(PROJECT_ROOT)),
            "dst_duplicate_diagnostics": str(DEFAULT_DST_DUPLICATE_OUTPUT.relative_to(PROJECT_ROOT)),
        },
        "methodological_notes": [
            "75-minute spring DST intervals are retained as single measurement intervals using van as interval_start.",
            "75-minute spring DST intervals are not split into artificial 15-minute rows and counts are not divided.",
            "Missing intervals are not imputed or treated as zero.",
            "sites.interval is not used as a filter.",
            "Absolute volume is not prepared as a clustering feature in this preprocessing step.",
        ],
        "file_errors": file_errors,
    }


def write_csv_safely(data: pd.DataFrame, path: Path) -> None:
    """Write CSV warnings with headers even when the frame is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def add_counter(counter: dict[str, int], values: pd.Series) -> None:
    """Accumulate value counts into a plain dict."""
    for key, value in values.value_counts(dropna=False).to_dict().items():
        counter[str(key)] = counter.get(str(key), 0) + int(value)


def append_warning_frame(frames: list[pd.DataFrame], data: pd.DataFrame) -> None:
    """Append a warning frame only when it contains rows."""
    if not data.empty:
        frames.append(warning_frame(data))


def combine_warning_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine warning frames while preserving headers for empty outputs."""
    if not frames:
        return pd.DataFrame(columns=WARNING_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def combine_duplicate_reports(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-file duplicate diagnostics with stable headers."""
    if not frames:
        return pd.DataFrame(columns=DUPLICATE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    for column in DUPLICATE_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined[DUPLICATE_COLUMNS].sort_values(
        ["date", "site_id", "richting", "duplicate_type"], na_position="last"
    ).reset_index(drop=True)


def run_preprocessing(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_output: Path = DEFAULT_PROCESSED_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    anomalous_output: Path = DEFAULT_ANOMALOUS_OUTPUT,
    negative_output: Path = DEFAULT_NEGATIVE_OUTPUT,
    duplicate_output: Path = DEFAULT_DUPLICATE_OUTPUT,
    dst_duplicate_output: Path = DEFAULT_DST_DUPLICATE_OUTPUT,
) -> dict[str, object]:
    """Run the full raw-to-site-level preprocessing step."""
    raw_dir = resolve_raw_dir(raw_dir)
    files = monthly_count_files(raw_dir)
    if not files:
        raise DataInspectionError(f"No monthly count CSV files found in {raw_dir}")

    # The monthly files are processed one by one so that warnings can still be
    # written even if one source file has a problem.
    metadata = load_metadata(raw_dir)
    processed_frames: list[pd.DataFrame] = []
    anomalous_frames: list[pd.DataFrame] = []
    negative_frames: list[pd.DataFrame] = []
    duplicate_frames: list[pd.DataFrame] = []
    file_errors: list[str] = []
    loaded_rows = 0
    filtered_rows = 0
    invalid_minute_count = 0
    kept_before_aggregation = 0
    duration_counts_kept: dict[str, int] = {}
    interval_status_counts_kept: dict[str, int] = {}
    minute_values_kept: set[int] = set()

    for index, path in enumerate(files, start=1):
        try:
            # 1. Read one month and keep only cyclist counts in the study period.
            raw = load_count_file(path)
            loaded_rows += len(raw)
            filtered = normalize_and_filter_counts(raw)
            filtered_rows += len(filtered)
            if filtered.empty:
                print(f"[{index}/{len(files)}] {path.name}: no rows after date/type filter", flush=True)
                continue

            # 2. Add time variables and separate warning rows from usable rows.
            filtered = add_time_columns(filtered)
            anomalous = filtered.loc[~filtered["interval_status"].isin(VALID_INTERVAL_STATUSES)].copy()
            append_warning_frame(anomalous_frames, anomalous)
            valid = filtered.loc[filtered["interval_status"].isin(VALID_INTERVAL_STATUSES)].copy()

            invalid_minutes = valid.loc[~valid["minute"].isin(VALID_MINUTES)].copy()
            invalid_minute_count += len(invalid_minutes)
            if not invalid_minutes.empty:
                valid = valid.loc[valid["minute"].isin(VALID_MINUTES)].copy()

            # 3. Negative counts are not used in the main time series.
            negative = valid.loc[valid["aantal"].lt(0)].copy()
            append_warning_frame(negative_frames, negative)
            if not negative.empty:
                valid = valid.loc[~valid["aantal"].lt(0)].copy()

            if valid.empty:
                print(f"[{index}/{len(files)}] {path.name}: no valid rows after warnings", flush=True)
                continue

            kept_before_aggregation += len(valid)
            add_counter(duration_counts_kept, valid["duration_minutes"])
            add_counter(interval_status_counts_kept, valid["interval_status"])
            minute_values_kept.update(int(value) for value in valid["minute"].dropna().unique().tolist())

            # 4. Keep duplicate diagnostics before summing directions.
            duplicate = build_duplicate_diagnostics(valid)
            if not duplicate.empty:
                duplicate_frames.append(duplicate)

            # 5. Sum directions to obtain one site-level count per timestamp.
            processed_frames.append(aggregate_site_level(valid))
            print(
                f"[{index}/{len(files)}] {path.name}: filtered={len(filtered):,}, valid={len(valid):,}, "
                f"anomalous={len(anomalous):,}, negative={len(negative):,}, duplicate_groups={len(duplicate):,}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep processing other files
            file_errors.append(f"{path.relative_to(PROJECT_ROOT)}: {type(exc).__name__}: {exc}")

    if not processed_frames:
        raise DataInspectionError("No valid FIETSERS rows found in selected date range")

    # A final groupby is a safety net in case a site-month boundary creates
    # duplicate site timestamps after the per-file processing.
    processed = pd.concat(processed_frames, ignore_index=True)
    if bool(processed.duplicated(["site_id", "interval_start"]).any()):
        processed = (
            processed.groupby(["site_id", "interval_start"], as_index=False, observed=True, dropna=False)
            .agg(
                count=("count", "sum"),
                date=("date", "first"),
                year=("year", "first"),
                month=("month", "first"),
                day_of_week=("day_of_week", "first"),
                is_weekend=("is_weekend", "first"),
                hour=("hour", "first"),
                minute=("minute", "first"),
                quarter_index=("quarter_index", "first"),
                has_dst_spring_75min=("has_dst_spring_75min", "max"),
                n_direction_rows=("n_direction_rows", "sum"),
                n_directions=("n_directions", "max"),
                duration_minutes_min=("duration_minutes_min", "min"),
                duration_minutes_max=("duration_minutes_max", "max"),
            )
            .reset_index(drop=True)
        )
        processed["interval_status"] = "normal_15min"
        processed.loc[processed["has_dst_spring_75min"], "interval_status"] = "dst_spring_75min"
        processed = processed[PROCESSED_COLUMNS]
    processed = processed.sort_values(["site_id", "interval_start"]).reset_index(drop=True)

    anomalous_rows = combine_warning_frames(anomalous_frames)
    negative_rows = combine_warning_frames(negative_frames)
    duplicate_report = combine_duplicate_reports(duplicate_frames)

    processed_output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(processed_output, index=False)
    write_csv_safely(anomalous_rows, anomalous_output)
    write_csv_safely(negative_rows, negative_output)
    write_csv_safely(duplicate_report, duplicate_output)

    report = build_report(
        raw_dir=raw_dir,
        files=files,
        metadata=metadata,
        loaded_rows=loaded_rows,
        filtered_rows=filtered_rows,
        anomalous_rows=anomalous_rows,
        negative_rows=negative_rows,
        invalid_minute_rows=invalid_minute_count,
        duplicate_report=duplicate_report,
        rows_kept_before_aggregation=kept_before_aggregation,
        processed=processed,
        duration_counts_kept=duration_counts_kept,
        interval_status_counts_kept=interval_status_counts_kept,
        minute_values_kept=minute_values_kept,
        file_errors=file_errors,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    dst_duplicate_payload = {
        "generated_at": report["generated_at"],
        "autumn_dst_transition_dates_europe_brussels": report["autumn_dst_transition_dates_europe_brussels"],
        "duplicate_diagnostics": report["duplicate_diagnostics"],
        "interpretation": "Duplicate local interval_start rows on the autumn DST repeated hour are likely clock-change effects; duplicates outside those timestamps are suspicious.",
    }
    dst_duplicate_output.parent.mkdir(parents=True, exist_ok=True)
    dst_duplicate_output.write_text(json.dumps(dst_duplicate_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess AWV raw cyclist counts to site-level interval parquet.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing raw AWV CSV files.")
    parser.add_argument("--processed-output", type=Path, default=DEFAULT_PROCESSED_OUTPUT, help="Processed parquet output path.")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT, help="JSON preprocessing report output path.")
    parser.add_argument("--anomalous-output", type=Path, default=DEFAULT_ANOMALOUS_OUTPUT, help="Anomalous duration CSV output path.")
    parser.add_argument("--negative-output", type=Path, default=DEFAULT_NEGATIVE_OUTPUT, help="Negative count CSV output path.")
    parser.add_argument("--duplicate-output", type=Path, default=DEFAULT_DUPLICATE_OUTPUT, help="Duplicate interval-start CSV output path.")
    parser.add_argument("--dst-duplicate-output", type=Path, default=DEFAULT_DST_DUPLICATE_OUTPUT, help="DST duplicate diagnostics JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_preprocessing(
            raw_dir=args.raw_dir,
            processed_output=args.processed_output,
            report_output=args.report_output,
            anomalous_output=args.anomalous_output,
            negative_output=args.negative_output,
            duplicate_output=args.duplicate_output,
            dst_duplicate_output=args.dst_duplicate_output,
        )
    except DataInspectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Preprocessing complete")
    print(f"- monthly files loaded: {report['monthly_files_loaded']}")
    print(f"- rows after date/type filter: {report['rows_after_date_and_type_filter']}")
    print(f"- anomalous duration rows: {report['anomalous_duration_rows']}")
    print(f"- negative count rows: {report['negative_count_rows']}")
    print(f"- invalid minute rows: {report['invalid_minute_rows']}")
    print(f"- rows kept before aggregation: {report['rows_kept_before_direction_aggregation']}")
    print(f"- processed site-interval rows: {report['processed_site_interval_rows']}")
    print(f"- duplicate diagnostics: {report['duplicate_diagnostics']}")
    print(f"- outputs: {report['outputs']}")
    if report["file_errors"]:
        print("File errors:")
        for error in report["file_errors"]:
            print(f"  - {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
