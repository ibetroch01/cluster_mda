"""Diagnose AWV count interval durations for bicycle rows.

This script is diagnostic only. It does not delete, correct, or aggregate any
rows. It checks whether non-15-minute count intervals are explainable by the
Europe/Brussels spring daylight saving time transition.
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

from src.inspect_data import (  # noqa: E402
    DataInspectionError,
    DEFAULT_RAW_DIR,
    PROJECT_ROOT,
    classify_file,
    detect_schema,
    read_csv_with_schema,
    resolve_raw_dir,
)

DEFAULT_YEARS = (2023, 2024, 2025)
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "outputs" / "interval_duration_summary.csv"
DEFAULT_NON_15_OUTPUT = PROJECT_ROOT / "outputs" / "non_15min_intervals.csv"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "interval_diagnostics_report.json"
MONTHLY_COUNT_PATTERN = re.compile(r"^data-(\d{4})-(\d{2})\.csv$", re.IGNORECASE)
BRUSSELS_TZ = ZoneInfo("Europe/Brussels")
REQUIRED_COLUMNS = ["site_id", "richting", "type", "van", "tot", "aantal"]
# Non-15-minute rows are listed explicitly because they drive the later cleaning
# decision about DST rows versus anomalous timing.
NON_15_COLUMNS = [
    "date",
    "year",
    "month",
    "van",
    "tot",
    "site_id",
    "richting",
    "aantal",
    "duration_minutes",
    "interval_status",
    "source_file",
]


def parse_year_month(path: Path) -> tuple[int, int] | None:
    """Parse YYYY-MM from an AWV monthly count filename."""
    match = MONTHLY_COUNT_PATTERN.match(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def monthly_count_files(raw_dir: Path, years: tuple[int, ...]) -> list[Path]:
    """Return monthly count CSV files whose filename year is in years."""
    selected: list[Path] = []
    for path in sorted(raw_dir.glob("data-*.csv")):
        parsed = parse_year_month(path)
        if parsed is None:
            continue
        year, _month = parsed
        if year in years:
            selected.append(path)
    return selected


def last_sunday(year: int, month: int) -> pd.Timestamp:
    """Return the date of the last Sunday for a year/month."""
    date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    while date.weekday() != 6:
        date -= pd.Timedelta(days=1)
    return date.normalize()


def brussels_spring_dst_date(year: int) -> pd.Timestamp:
    """Return the Europe/Brussels spring DST transition date for a year.

    The EU spring transition is the last Sunday in March. This function keeps
    the timezone dependency explicit for documentation and future refinement.
    """
    transition_date = last_sunday(year, 3)
    # Validate the local offset changes on this date in Europe/Brussels.
    before = pd.Timestamp(year=year, month=3, day=transition_date.day, hour=1, tz=BRUSSELS_TZ).utcoffset()
    after = pd.Timestamp(year=year, month=3, day=transition_date.day, hour=4, tz=BRUSSELS_TZ).utcoffset()
    if before is None or after is None or after <= before:
        raise DataInspectionError(f"Could not validate Brussels spring DST transition for {year}")
    return transition_date


def is_brussels_spring_dst_75(van: pd.Series, duration_minutes: pd.Series) -> pd.Series:
    """Return True for 75-minute rows on the Brussels spring DST transition date."""
    dates = van.dt.normalize()
    transition_dates = {year: brussels_spring_dst_date(int(year)) for year in sorted(van.dt.year.dropna().unique())}
    expected_dates = van.dt.year.map(transition_dates)
    return duration_minutes.eq(75.0) & dates.eq(expected_dates)


def load_fietsers_counts(path: Path) -> pd.DataFrame:
    """Load one monthly count file and return FIETSERS rows with parsed intervals."""
    if classify_file(path) != "counts":
        raise DataInspectionError(f"Not a monthly count CSV: {path}")
    schema = detect_schema(path, path.parent)
    data = read_csv_with_schema(path, schema, columns=REQUIRED_COLUMNS)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise DataInspectionError(f"{path.name} missing columns after normalization: {', '.join(missing)}")

    data = data.copy()
    data["source_file"] = str(path.relative_to(PROJECT_ROOT))
    type_text = data["type"].astype("string").str.strip().str.casefold()
    data = data.loc[type_text == "fietsers"].copy()
    data["van_dt"] = pd.to_datetime(data["van"], errors="coerce")
    data["tot_dt"] = pd.to_datetime(data["tot"], errors="coerce")
    data["duration_minutes"] = (data["tot_dt"] - data["van_dt"]).dt.total_seconds() / 60.0
    data["date"] = data["van_dt"].dt.date.astype("string")
    data["year"] = data["van_dt"].dt.year.astype("Int64")
    data["month"] = data["van_dt"].dt.month.astype("Int64")
    data["aantal"] = pd.to_numeric(data["aantal"], errors="coerce")
    return data


def add_interval_status(data: pd.DataFrame) -> pd.DataFrame:
    """Add interval_status according to diagnostic duration rules."""
    data = data.copy()
    # Start from the strictest status and only promote rows when the duration
    # matches the known normal or DST pattern.
    data["interval_status"] = "anomalous_duration"
    data.loc[data["duration_minutes"].eq(15.0), "interval_status"] = "normal_15min"
    valid_van = data["van_dt"].notna() & data["duration_minutes"].notna()
    dst_75 = pd.Series(False, index=data.index)
    if valid_van.any():
        dst_75.loc[valid_van] = is_brussels_spring_dst_75(data.loc[valid_van, "van_dt"], data.loc[valid_van, "duration_minutes"])
    data.loc[dst_75, "interval_status"] = "dst_spring_75min"
    return data


def build_duration_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count all unique duration values and interval statuses."""
    summary = (
        data.groupby(["duration_minutes", "interval_status"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
        .sort_values(["duration_minutes", "interval_status"], na_position="last")
        .reset_index(drop=True)
    )
    return summary


def build_non_15_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Return the requested row-level listing for non-15-minute intervals."""
    non_15 = data.loc[~data["duration_minutes"].eq(15.0)].copy()
    for column in NON_15_COLUMNS:
        if column not in non_15.columns:
            non_15[column] = pd.NA
    return non_15[NON_15_COLUMNS].sort_values(["van", "site_id", "richting"]).reset_index(drop=True)


def report_payload(
    raw_dir: Path,
    years: tuple[int, ...],
    files: list[Path],
    data: pd.DataFrame,
    duration_summary: pd.DataFrame,
    non_15: pd.DataFrame,
    file_errors: list[str],
) -> dict[str, object]:
    """Create a machine-readable diagnostics report."""
    transition_dates = {str(year): brussels_spring_dst_date(year).date().isoformat() for year in years}
    status_counts = data["interval_status"].value_counts(dropna=False).to_dict() if not data.empty else {}
    duration_counts = {
        str(row.duration_minutes): int(row.row_count)
        for row in duration_summary.groupby("duration_minutes", dropna=False, as_index=False)["row_count"].sum().itertuples(index=False)
    }
    rows_75 = data.loc[data["duration_minutes"].eq(75.0)]
    rows_75_on_dst = rows_75.loc[rows_75["interval_status"].eq("dst_spring_75min")]
    rows_75_only_dst = len(rows_75) == len(rows_75_on_dst)
    anomalous = data.loc[data["interval_status"].eq("anomalous_duration")]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
        "years": list(years),
        "n_files": len(files),
        "files": [str(path.relative_to(PROJECT_ROOT)) for path in files],
        "n_fietsers_rows": int(len(data)),
        "duration_minutes_counts": duration_counts,
        "interval_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "spring_dst_transition_dates_europe_brussels": transition_dates,
        "duration_75_occurs_only_on_spring_dst_dates": bool(rows_75_only_dst),
        "n_duration_75_rows": int(len(rows_75)),
        "n_non_15min_rows": int(len(non_15)),
        "n_anomalous_duration_rows": int(len(anomalous)),
        "outputs": {
            "duration_summary": str(DEFAULT_SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)),
            "non_15min_intervals": str(DEFAULT_NON_15_OUTPUT.relative_to(PROJECT_ROOT)),
            "report": str(DEFAULT_REPORT_OUTPUT.relative_to(PROJECT_ROOT)),
        },
        "file_errors": file_errors,
    }


def run_diagnostics(
    raw_dir: Path = DEFAULT_RAW_DIR,
    years: tuple[int, ...] = DEFAULT_YEARS,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    non_15_output: Path = DEFAULT_NON_15_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, object]:
    """Run interval diagnostics and write all outputs."""
    raw_dir = resolve_raw_dir(raw_dir)
    files = monthly_count_files(raw_dir, years)
    if not files:
        raise DataInspectionError(f"No monthly count CSV files found in {raw_dir} for years {years}")

    frames: list[pd.DataFrame] = []
    file_errors: list[str] = []
    for path in files:
        try:
            frames.append(load_fietsers_counts(path))
        except Exception as exc:  # noqa: BLE001 - continue diagnostics across files
            file_errors.append(f"{path.relative_to(PROJECT_ROOT)}: {type(exc).__name__}: {exc}")

    if frames:
        data = pd.concat(frames, ignore_index=True)
    else:
        raise DataInspectionError("No FIETSERS rows could be loaded from selected files")

    data = add_interval_status(data)
    duration_summary = build_duration_summary(data)
    non_15 = build_non_15_intervals(data)
    report = report_payload(raw_dir, years, files, data, duration_summary, non_15, file_errors)

    for path in [summary_output, non_15_output, report_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    duration_summary.to_csv(summary_output, index=False)
    non_15.to_csv(non_15_output, index=False)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_years(value: str) -> tuple[int, ...]:
    """Parse a comma-separated year list."""
    years = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not years:
        raise argparse.ArgumentTypeError("At least one year is required")
    return years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose AWV count interval durations for FIETSERS rows.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing raw AWV monthly CSV files.")
    parser.add_argument("--years", type=parse_years, default=DEFAULT_YEARS, help="Comma-separated years to include, default: 2023,2024,2025.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT, help="Duration summary CSV output path.")
    parser.add_argument("--non-15-output", type=Path, default=DEFAULT_NON_15_OUTPUT, help="Non-15-minute row listing CSV output path.")
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT, help="JSON diagnostics report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_diagnostics(
            raw_dir=args.raw_dir,
            years=args.years,
            summary_output=args.summary_output,
            non_15_output=args.non_15_output,
            report_output=args.report_output,
        )
    except DataInspectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Interval diagnostics complete")
    print(f"- files: {report['n_files']}")
    print(f"- FIETSERS rows: {report['n_fietsers_rows']}")
    print(f"- duration counts: {report['duration_minutes_counts']}")
    print(f"- interval status counts: {report['interval_status_counts']}")
    print(f"- 75-minute intervals only on Brussels spring DST dates: {report['duration_75_occurs_only_on_spring_dst_dates']}")
    print(f"- outputs: {report['outputs']}")
    if report["file_errors"]:
        print("File errors:")
        for error in report["file_errors"]:
            print(f"  - {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
