"""Build location-group-level data quality reports for AWV cyclist counts.

Coverage and volume variables produced here are for filtering and diagnostics
only. They must not be used as KMeans clustering features.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "counts_location_group_2023_2025.parquet"
DEFAULT_DAY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "location_group_day_quality.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "location_group_quality_summary.csv"
DEFAULT_ELIGIBLE_OUTPUT = PROJECT_ROOT / "data" / "processed" / "eligible_location_groups.csv"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "location_group_quality_report.json"
NORMAL_DAY_INTERVALS = 96
VALID_DAY_COVERAGE_THRESHOLD = 0.90
QUALITY_COLUMNS = [
    "n_valid_days",
    "n_valid_weekdays",
    "n_valid_weekend_days",
    "n_valid_summer_days",
    "n_valid_winter_days",
    "n_months_with_valid_data",
    "n_years_with_valid_data",
    "median_daily_total",
    "mean_daily_total",
    "percentage_full_coverage_intervals",
    "percentage_partial_coverage_intervals",
    "morning_07_09_coverage_ratio",
    "evening_16_18_coverage_ratio",
    "weekday_midday_10_15_coverage_ratio",
    "weekend_midday_afternoon_11_17_coverage_ratio",
]


@dataclass(frozen=True)
class LocationQualityThresholds:
    """Candidate location-group quality thresholds."""

    min_valid_days: int = 365
    min_valid_weekdays: int = 240
    min_valid_weekend_days: int = 90
    min_valid_summer_days: int = 50
    min_valid_winter_days: int = 50
    min_months_with_valid_data: int = 12
    min_years_with_valid_data: int = 2


def load_location_counts(path: Path) -> pd.DataFrame:
    """Load location-group counts and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Location-group counts parquet not found: {path}")
    data = pd.read_parquet(path)
    required = {
        "location_group_id",
        "interval_start",
        "count",
        "n_sites_present",
        "n_sites_expected",
        "interval_full_coverage",
        "date",
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        "hour",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Location counts missing required columns: {', '.join(missing)}")
    data = data.copy()
    data["interval_start"] = pd.to_datetime(data["interval_start"])
    data["date"] = pd.to_datetime(data["date"]).dt.date
    data["count"] = pd.to_numeric(data["count"], errors="coerce").fillna(0)
    data["n_sites_present"] = pd.to_numeric(data["n_sites_present"], errors="coerce").fillna(0)
    data["n_sites_expected"] = pd.to_numeric(data["n_sites_expected"], errors="coerce").fillna(0)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data["month"] = pd.to_numeric(data["month"], errors="coerce").astype("Int64")
    data["day_of_week"] = pd.to_numeric(data["day_of_week"], errors="coerce").astype("Int64")
    data["hour"] = pd.to_numeric(data["hour"], errors="coerce").astype("Int64")
    data["is_weekend"] = data["day_of_week"].isin([5, 6])
    data["interval_full_coverage"] = data["interval_full_coverage"].astype(bool)
    return data


def coverage_ratio(observed: float, expected: float) -> float:
    """Safely divide observed by expected coverage counts."""
    if pd.isna(expected) or float(expected) <= 0:
        return 0.0
    return float(observed) / float(expected)


def build_location_group_day_quality(
    counts: pd.DataFrame,
    valid_day_coverage_threshold: float = VALID_DAY_COVERAGE_THRESHOLD,
) -> pd.DataFrame:
    """Compute per-location-group per-date site-interval coverage."""
    day = (
        counts.groupby(["location_group_id", "date"], as_index=False, observed=True)
        .agg(
            n_sites_expected=("n_sites_expected", "max"),
            n_group_intervals_present=("interval_start", "nunique"),
            n_site_interval_observations_present=("n_sites_present", "sum"),
            daily_total=("count", "sum"),
            year=("year", "first"),
            month=("month", "first"),
            day_of_week=("day_of_week", "first"),
            is_weekend=("is_weekend", "first"),
        )
        .sort_values(["location_group_id", "date"])
        .reset_index(drop=True)
    )
    day["expected_site_interval_observations"] = NORMAL_DAY_INTERVALS * day["n_sites_expected"]
    day["site_interval_coverage_ratio"] = [
        coverage_ratio(observed, expected)
        for observed, expected in zip(day["n_site_interval_observations_present"], day["expected_site_interval_observations"])
    ]
    day["valid_day"] = day["site_interval_coverage_ratio"] >= valid_day_coverage_threshold
    day["date"] = day["date"].astype(str)
    return day[
        [
            "location_group_id",
            "date",
            "n_sites_expected",
            "n_group_intervals_present",
            "n_site_interval_observations_present",
            "expected_site_interval_observations",
            "site_interval_coverage_ratio",
            "daily_total",
            "valid_day",
            "year",
            "month",
            "day_of_week",
            "is_weekend",
        ]
    ]


def window_coverage(counts: pd.DataFrame, mask: pd.Series, column_name: str) -> pd.DataFrame:
    """Compute location-level site-interval coverage for a temporal window."""
    subset = counts.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame({"location_group_id": counts["location_group_id"].drop_duplicates(), column_name: 0.0})
    grouped = subset.groupby("location_group_id", as_index=False, observed=True).agg(
        observed=("n_sites_present", "sum"),
        intervals=("interval_start", "nunique"),
        n_sites_expected=("n_sites_expected", "max"),
    )
    grouped["expected"] = grouped["intervals"] * grouped["n_sites_expected"]
    grouped[column_name] = [coverage_ratio(obs, exp) for obs, exp in zip(grouped["observed"], grouped["expected"])]
    all_groups = pd.DataFrame({"location_group_id": counts["location_group_id"].drop_duplicates()})
    return all_groups.merge(grouped[["location_group_id", column_name]], on="location_group_id", how="left").fillna({column_name: 0.0})


def build_window_coverage_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Build location-level coverage ratios for key feature-engineering windows."""
    weekday = ~counts["is_weekend"].astype(bool)
    weekend = counts["is_weekend"].astype(bool)
    windows = [
        window_coverage(counts, counts["hour"].between(7, 9), "morning_07_09_coverage_ratio"),
        window_coverage(counts, counts["hour"].between(16, 18), "evening_16_18_coverage_ratio"),
        window_coverage(counts, weekday & counts["hour"].between(10, 15), "weekday_midday_10_15_coverage_ratio"),
        window_coverage(counts, weekend & counts["hour"].between(11, 17), "weekend_midday_afternoon_11_17_coverage_ratio"),
    ]
    output = windows[0]
    for frame in windows[1:]:
        output = output.merge(frame, on="location_group_id", how="outer")
    return output.fillna(0.0)


def build_location_group_quality_summary(day_quality: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily quality indicators to one row per location group."""
    day = day_quality.copy()
    day["date_dt"] = pd.to_datetime(day["date"])
    day["year_month"] = day["date_dt"].dt.to_period("M").astype(str)
    day["is_summer"] = day["month"].isin([6, 7, 8])
    day["is_winter"] = day["month"].isin([12, 1, 2])
    valid = day.loc[day["valid_day"]].copy()

    base = (
        day.groupby("location_group_id", as_index=False, observed=True)
        .agg(
            median_daily_total=("daily_total", "median"),
            mean_daily_total=("daily_total", "mean"),
        )
    )
    interval_coverage = (
        counts.groupby("location_group_id", as_index=False, observed=True)
        .agg(
            total_intervals=("interval_start", "size"),
            full_coverage_intervals=("interval_full_coverage", lambda values: int(values.astype(bool).sum())),
        )
    )
    interval_coverage["partial_coverage_intervals"] = interval_coverage["total_intervals"] - interval_coverage["full_coverage_intervals"]
    interval_coverage["percentage_full_coverage_intervals"] = interval_coverage["full_coverage_intervals"] / interval_coverage["total_intervals"]
    interval_coverage["percentage_partial_coverage_intervals"] = interval_coverage["partial_coverage_intervals"] / interval_coverage["total_intervals"]

    if valid.empty:
        valid_summary = pd.DataFrame({"location_group_id": base["location_group_id"]})
    else:
        valid_summary = (
            valid.groupby("location_group_id", as_index=False, observed=True)
            .agg(
                n_valid_days=("date", "nunique"),
                n_valid_weekdays=("is_weekend", lambda values: int((~values.astype(bool)).sum())),
                n_valid_weekend_days=("is_weekend", lambda values: int(values.astype(bool).sum())),
                n_valid_summer_days=("is_summer", lambda values: int(values.astype(bool).sum())),
                n_valid_winter_days=("is_winter", lambda values: int(values.astype(bool).sum())),
                n_months_with_valid_data=("year_month", "nunique"),
                n_years_with_valid_data=("year", "nunique"),
                first_valid_date=("date_dt", "min"),
                last_valid_date=("date_dt", "max"),
            )
        )

    summary = base.merge(valid_summary, on="location_group_id", how="left")
    summary = summary.merge(
        interval_coverage[
            [
                "location_group_id",
                "percentage_full_coverage_intervals",
                "percentage_partial_coverage_intervals",
            ]
        ],
        on="location_group_id",
        how="left",
    )
    summary = summary.merge(build_window_coverage_summary(counts), on="location_group_id", how="left")
    fill_zero = [
        "n_valid_days",
        "n_valid_weekdays",
        "n_valid_weekend_days",
        "n_valid_summer_days",
        "n_valid_winter_days",
        "n_months_with_valid_data",
        "n_years_with_valid_data",
    ]
    for column in fill_zero:
        if column not in summary.columns:
            summary[column] = 0
        summary[column] = summary[column].fillna(0).astype(int)
    for column in ["first_valid_date", "last_valid_date"]:
        if column not in summary.columns:
            summary[column] = pd.NaT
        summary[column] = pd.to_datetime(summary[column], errors="coerce").dt.date.astype("string")
    ordered = [
        "location_group_id",
        "n_valid_days",
        "n_valid_weekdays",
        "n_valid_weekend_days",
        "n_valid_summer_days",
        "n_valid_winter_days",
        "n_months_with_valid_data",
        "n_years_with_valid_data",
        "first_valid_date",
        "last_valid_date",
        "median_daily_total",
        "mean_daily_total",
        "percentage_full_coverage_intervals",
        "percentage_partial_coverage_intervals",
        "morning_07_09_coverage_ratio",
        "evening_16_18_coverage_ratio",
        "weekday_midday_10_15_coverage_ratio",
        "weekend_midday_afternoon_11_17_coverage_ratio",
    ]
    return summary[ordered].sort_values("location_group_id").reset_index(drop=True)


def apply_candidate_thresholds(summary: pd.DataFrame, thresholds: LocationQualityThresholds) -> pd.DataFrame:
    """Add threshold pass/fail columns and final candidate eligibility."""
    threshold_map = asdict(thresholds)
    eligible = summary.copy()
    checks = {
        "passes_min_valid_days": ("n_valid_days", threshold_map["min_valid_days"]),
        "passes_min_valid_weekdays": ("n_valid_weekdays", threshold_map["min_valid_weekdays"]),
        "passes_min_valid_weekend_days": ("n_valid_weekend_days", threshold_map["min_valid_weekend_days"]),
        "passes_min_valid_summer_days": ("n_valid_summer_days", threshold_map["min_valid_summer_days"]),
        "passes_min_valid_winter_days": ("n_valid_winter_days", threshold_map["min_valid_winter_days"]),
        "passes_min_months_with_valid_data": ("n_months_with_valid_data", threshold_map["min_months_with_valid_data"]),
        "passes_min_years_with_valid_data": ("n_years_with_valid_data", threshold_map["min_years_with_valid_data"]),
    }
    pass_columns = []
    for pass_column, (source_column, threshold) in checks.items():
        eligible[pass_column] = eligible[source_column] >= threshold
        pass_columns.append(pass_column)
    eligible["eligible_location_group"] = eligible[pass_columns].all(axis=1)
    return eligible


def threshold_step_counts(summary: pd.DataFrame, thresholds: LocationQualityThresholds) -> list[dict[str, object]]:
    """Return number of groups remaining after each threshold is applied sequentially."""
    steps = [
        ("min_valid_days", "n_valid_days", thresholds.min_valid_days),
        ("min_valid_weekdays", "n_valid_weekdays", thresholds.min_valid_weekdays),
        ("min_valid_weekend_days", "n_valid_weekend_days", thresholds.min_valid_weekend_days),
        ("min_valid_summer_days", "n_valid_summer_days", thresholds.min_valid_summer_days),
        ("min_valid_winter_days", "n_valid_winter_days", thresholds.min_valid_winter_days),
        ("min_months_with_valid_data", "n_months_with_valid_data", thresholds.min_months_with_valid_data),
        ("min_years_with_valid_data", "n_years_with_valid_data", thresholds.min_years_with_valid_data),
    ]
    remaining = pd.Series(True, index=summary.index)
    output: list[dict[str, object]] = []
    for name, column, threshold in steps:
        remaining &= summary[column] >= threshold
        output.append(
            {
                "threshold": name,
                "column": column,
                "minimum": threshold,
                "n_location_groups_remaining": int(remaining.sum()),
                "n_location_groups_removed_cumulative": int(len(summary) - remaining.sum()),
            }
        )
    return output


def distribution_summaries(summary: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Create compact distributions for all quality variables."""
    distributions: dict[str, dict[str, float | int | None]] = {}
    for column in QUALITY_COLUMNS:
        values = pd.to_numeric(summary[column], errors="coerce")
        if values.dropna().empty:
            distributions[column] = {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}
            continue
        distributions[column] = {
            "count": int(values.notna().sum()),
            "min": float(values.min()),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "mean": float(values.mean()),
            "p75": float(values.quantile(0.75)),
            "max": float(values.max()),
        }
    return distributions


def build_report(
    day_quality: pd.DataFrame,
    summary: pd.DataFrame,
    eligible: pd.DataFrame,
    thresholds: LocationQualityThresholds,
    valid_day_threshold: float,
) -> dict[str, object]:
    """Build machine-readable location-group quality report."""
    n_before = int(len(summary))
    n_after = int(eligible["eligible_location_group"].sum())
    removal_share = (n_before - n_after) / n_before if n_before else 0.0
    warnings: list[str] = []
    if removal_share >= 0.5:
        warnings.append(f"Candidate thresholds remove {removal_share:.1%} of location groups; review before using as final filter.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(DEFAULT_INPUT.relative_to(PROJECT_ROOT)),
        "valid_day_definition": {
            "normal_day_intervals_per_site": NORMAL_DAY_INTERVALS,
            "expected_site_interval_observations": "96 * n_sites_expected",
            "site_interval_coverage_ratio_minimum": valid_day_threshold,
        },
        "thresholds_candidate": asdict(thresholds),
        "n_location_groups_before_filtering": n_before,
        "n_location_groups_after_all_thresholds": n_after,
        "n_location_groups_removed": int(n_before - n_after),
        "share_location_groups_removed": removal_share,
        "location_groups_after_each_threshold": threshold_step_counts(summary, thresholds),
        "day_quality_rows": int(len(day_quality)),
        "valid_day_distribution": {str(key): int(value) for key, value in day_quality["valid_day"].value_counts(dropna=False).to_dict().items()},
        "distribution_summaries": distribution_summaries(summary),
        "warnings": warnings,
        "methodological_notes": [
            "Quality variables and volume variables are for filtering and diagnostics only.",
            "They must not enter the KMeans clustering feature matrix.",
            "Missing intervals are not imputed as zero; coverage is based on observed site-interval rows.",
        ],
        "outputs": {
            "location_group_day_quality": str(DEFAULT_DAY_OUTPUT.relative_to(PROJECT_ROOT)),
            "location_group_quality_summary": str(DEFAULT_SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)),
            "eligible_location_groups": str(DEFAULT_ELIGIBLE_OUTPUT.relative_to(PROJECT_ROOT)),
            "quality_report": str(DEFAULT_REPORT_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }


def run_location_group_quality_report(
    input_path: Path = DEFAULT_INPUT,
    day_output: Path = DEFAULT_DAY_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
    eligible_output: Path = DEFAULT_ELIGIBLE_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    thresholds: LocationQualityThresholds = LocationQualityThresholds(),
    valid_day_threshold: float = VALID_DAY_COVERAGE_THRESHOLD,
) -> dict[str, object]:
    """Build and write all location-group quality outputs."""
    counts = load_location_counts(input_path)
    day_quality = build_location_group_day_quality(counts, valid_day_coverage_threshold=valid_day_threshold)
    summary = build_location_group_quality_summary(day_quality, counts)
    eligible = apply_candidate_thresholds(summary, thresholds)
    report = build_report(day_quality, summary, eligible, thresholds, valid_day_threshold)

    day_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    day_quality.to_csv(day_output, index=False)
    summary.to_csv(summary_output, index=False)
    eligible.to_csv(eligible_output, index=False)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build location-group-level AWV count quality report.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--valid-day-coverage-threshold", type=float, default=VALID_DAY_COVERAGE_THRESHOLD)
    parser.add_argument("--min-valid-days", type=int, default=LocationQualityThresholds.min_valid_days)
    parser.add_argument("--min-valid-weekdays", type=int, default=LocationQualityThresholds.min_valid_weekdays)
    parser.add_argument("--min-valid-weekend-days", type=int, default=LocationQualityThresholds.min_valid_weekend_days)
    parser.add_argument("--min-valid-summer-days", type=int, default=LocationQualityThresholds.min_valid_summer_days)
    parser.add_argument("--min-valid-winter-days", type=int, default=LocationQualityThresholds.min_valid_winter_days)
    parser.add_argument("--min-months-with-valid-data", type=int, default=LocationQualityThresholds.min_months_with_valid_data)
    parser.add_argument("--min-years-with-valid-data", type=int, default=LocationQualityThresholds.min_years_with_valid_data)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = LocationQualityThresholds(
        min_valid_days=args.min_valid_days,
        min_valid_weekdays=args.min_valid_weekdays,
        min_valid_weekend_days=args.min_valid_weekend_days,
        min_valid_summer_days=args.min_valid_summer_days,
        min_valid_winter_days=args.min_valid_winter_days,
        min_months_with_valid_data=args.min_months_with_valid_data,
        min_years_with_valid_data=args.min_years_with_valid_data,
    )
    try:
        report = run_location_group_quality_report(
            input_path=args.input,
            thresholds=thresholds,
            valid_day_threshold=args.valid_day_coverage_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Location-group quality report complete")
    print(f"- groups before filtering: {report['n_location_groups_before_filtering']}")
    print(f"- groups after all candidate thresholds: {report['n_location_groups_after_all_thresholds']}")
    print(f"- share removed: {report['share_location_groups_removed']:.1%}")
    print("- groups after each threshold:")
    for step in report["location_groups_after_each_threshold"]:
        print(f"  {step['threshold']}: {step['n_location_groups_remaining']}")
    if report["warnings"]:
        print("- warnings:")
        for warning in report["warnings"]:
            print(f"  {warning}")
    print(f"- outputs: {report['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
