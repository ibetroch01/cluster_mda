"""Engineer candidate temporal features for location-group KMeans clustering.

The selected KMeans feature matrix intentionally contains only relative temporal
pattern features. Absolute volume, geography, identifiers, and data-quality
variables are kept as diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COUNTS_INPUT = PROJECT_ROOT / "data" / "processed" / "counts_location_group_2023_2025.parquet"
DEFAULT_DAY_QUALITY_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_day_quality.csv"
DEFAULT_ELIGIBLE_INPUT = PROJECT_ROOT / "data" / "processed" / "eligible_location_groups.csv"
DEFAULT_LOCATION_GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
DEFAULT_FEATURE_OUTPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_candidate.csv"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "location_group_feature_engineering_report.json"
WINDOW_COVERAGE_THRESHOLD = 0.90

SELECTED_KMEANS_FEATURES = [
    "log_weekend_weekday_ratio",
    "weekday_morning_peak_share",
    "weekday_evening_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "weekday_peak_concentration_2h",
    "log_summer_winter_ratio",
]
OPTIONAL_CANDIDATE_FEATURES = [
    "night_share",
    "daily_variability_cv",
]
FORBIDDEN_CLUSTERING_FEATURES = {
    "total_count",
    "avg_daily_count",
    "mean_daily_total",
    "median_daily_total",
    "max_daily_count",
    "n_valid_days",
    "n_valid_weekdays",
    "n_valid_weekend_days",
    "n_sites_in_group",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "mean_latitude",
    "mean_longitude",
    "gemeente",
    "gemeente_values",
    "site_id",
    "site_ids",
    "location_group_id",
}
DIAGNOSTIC_COLUMNS = [
    "n_valid_days",
    "n_valid_weekdays",
    "n_valid_weekend_days",
    "n_valid_summer_days",
    "n_valid_winter_days",
    "n_months_with_valid_data",
    "n_years_with_valid_data",
    "n_sites_in_group",
    "mean_daily_total",
    "median_daily_total",
    "percentage_full_coverage_intervals",
    "percentage_partial_coverage_intervals",
]


def load_inputs(
    counts_path: Path,
    day_quality_path: Path,
    eligible_path: Path,
    location_groups_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate required feature-engineering inputs."""
    for path in [counts_path, day_quality_path, eligible_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    counts = pd.read_parquet(counts_path)
    day_quality = pd.read_csv(day_quality_path)
    eligible = pd.read_csv(eligible_path)
    location_groups = pd.read_csv(location_groups_path) if location_groups_path.exists() else pd.DataFrame()

    required_counts = {"location_group_id", "interval_start", "count", "n_sites_present", "n_sites_expected", "hour", "quarter_index", "date"}
    required_day = {"location_group_id", "date", "daily_total", "valid_day", "is_weekend", "month", "day_of_week"}
    required_eligible = {"location_group_id", "eligible_location_group"}
    missing_counts = sorted(required_counts - set(counts.columns))
    missing_day = sorted(required_day - set(day_quality.columns))
    missing_eligible = sorted(required_eligible - set(eligible.columns))
    if missing_counts:
        raise ValueError(f"Counts input missing columns: {', '.join(missing_counts)}")
    if missing_day:
        raise ValueError(f"Day quality input missing columns: {', '.join(missing_day)}")
    if missing_eligible:
        raise ValueError(f"Eligible input missing columns: {', '.join(missing_eligible)}")

    counts = counts.copy()
    counts["interval_start"] = pd.to_datetime(counts["interval_start"])
    counts["date"] = pd.to_datetime(counts["date"]).dt.date.astype(str)
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0.0)
    counts["n_sites_present"] = pd.to_numeric(counts["n_sites_present"], errors="coerce").fillna(0.0)
    counts["n_sites_expected"] = pd.to_numeric(counts["n_sites_expected"], errors="coerce").fillna(0.0)
    counts["hour"] = pd.to_numeric(counts["hour"], errors="coerce")
    counts["quarter_index"] = pd.to_numeric(counts["quarter_index"], errors="coerce")

    day_quality = day_quality.copy()
    day_quality["date"] = pd.to_datetime(day_quality["date"]).dt.date.astype(str)
    day_quality["daily_total"] = pd.to_numeric(day_quality["daily_total"], errors="coerce").fillna(0.0)
    day_quality["valid_day"] = day_quality["valid_day"].astype(bool)
    day_quality["is_weekend"] = day_quality["is_weekend"].astype(bool)
    day_quality["month"] = pd.to_numeric(day_quality["month"], errors="coerce")
    day_quality["day_of_week"] = pd.to_numeric(day_quality["day_of_week"], errors="coerce")

    eligible = eligible.copy()
    eligible["eligible_location_group"] = eligible["eligible_location_group"].astype(bool)
    return counts, day_quality, eligible, location_groups


def window_daily_shares(
    counts: pd.DataFrame,
    valid_positive_days: pd.DataFrame,
    mask: pd.Series,
    output_column: str,
) -> pd.DataFrame:
    """Compute per-day count share for a fixed window with coverage filtering."""
    keys = ["location_group_id", "date"]
    daily = valid_positive_days[keys + ["daily_total", "is_weekend", "month", "day_of_week"]].copy()
    subset = counts.loc[mask].copy()
    if subset.empty:
        daily[output_column] = np.nan
        daily[f"{output_column}_coverage_ok"] = False
        return daily[keys + [output_column, f"{output_column}_coverage_ok"]]
    grouped = subset.groupby(keys, as_index=False, observed=True).agg(
        window_count=("count", "sum"),
        observed_site_intervals=("n_sites_present", "sum"),
        n_group_intervals=("interval_start", "nunique"),
        n_sites_expected=("n_sites_expected", "max"),
    )
    grouped["expected_site_intervals"] = grouped["n_group_intervals"] * grouped["n_sites_expected"]
    grouped["coverage_ratio"] = grouped["observed_site_intervals"] / grouped["expected_site_intervals"]
    output = daily.merge(grouped, on=keys, how="left")
    output["coverage_ratio"] = output["coverage_ratio"].fillna(0.0)
    output["window_count"] = output["window_count"].fillna(0.0)
    output[f"{output_column}_coverage_ok"] = output["coverage_ratio"] >= WINDOW_COVERAGE_THRESHOLD
    output[output_column] = np.where(
        output[f"{output_column}_coverage_ok"],
        output["window_count"] / output["daily_total"],
        np.nan,
    )
    return output[keys + [output_column, f"{output_column}_coverage_ok"]]


def rolling_2h_peak_daily_shares(counts: pd.DataFrame, valid_positive_days: pd.DataFrame) -> pd.DataFrame:
    """Compute max 2-hour daily share using a day x quarter pivot."""
    keys = ["location_group_id", "date"]
    counts = counts.copy()
    counts["quarter_index"] = pd.to_numeric(counts["quarter_index"], errors="coerce").astype("int64")
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0.0).astype("float64")
    counts["n_sites_present"] = pd.to_numeric(counts["n_sites_present"], errors="coerce").fillna(0.0).astype("float64")
    counts["expected_present"] = pd.to_numeric(counts["n_sites_expected"], errors="coerce").fillna(0.0).astype("float64")
    count_pivot = counts.pivot_table(index=keys, columns="quarter_index", values="count", aggfunc="sum", fill_value=0.0, sort=False)
    present_pivot = counts.pivot_table(index=keys, columns="quarter_index", values="n_sites_present", aggfunc="sum", fill_value=0.0, sort=False)
    expected_pivot = counts.pivot_table(index=keys, columns="quarter_index", values="expected_present", aggfunc="max", fill_value=0.0, sort=False)
    for quarter in range(96):
        if quarter not in count_pivot.columns:
            count_pivot[quarter] = 0.0
            present_pivot[quarter] = 0.0
            expected_pivot[quarter] = 0.0
    count_pivot = count_pivot[range(96)].sort_index(axis=1)
    present_pivot = present_pivot[range(96)].sort_index(axis=1)
    expected_pivot = expected_pivot[range(96)].sort_index(axis=1)

    count_roll = count_pivot.T.rolling(window=8).sum().T.iloc[:, 7:]
    present_roll = present_pivot.T.rolling(window=8).sum().T.iloc[:, 7:]
    expected_roll = expected_pivot.T.rolling(window=8).sum().T.iloc[:, 7:]
    coverage = present_roll / expected_roll.replace(0, np.nan)
    valid_window = coverage >= WINDOW_COVERAGE_THRESHOLD
    valid_counts = count_roll.where(valid_window)
    max_count = valid_counts.max(axis=1)
    n_windows = valid_window.sum(axis=1)
    output = valid_positive_days[keys + ["daily_total", "is_weekend"]].copy()
    output = output.merge(
        max_count.rename("weekday_peak_concentration_2h_count").reset_index(),
        on=keys,
        how="left",
    )
    output = output.merge(n_windows.rename("n_valid_rolling_2h_windows").reset_index(), on=keys, how="left")
    output["weekday_peak_concentration_2h_day"] = np.where(
        (~output["is_weekend"]) & output["weekday_peak_concentration_2h_count"].notna(),
        output["weekday_peak_concentration_2h_count"] / output["daily_total"],
        np.nan,
    )
    output.loc[output["is_weekend"], "n_valid_rolling_2h_windows"] = 0
    output["n_valid_rolling_2h_windows"] = output["n_valid_rolling_2h_windows"].fillna(0).astype(int)
    return output[keys + ["weekday_peak_concentration_2h_day", "n_valid_rolling_2h_windows"]]


def build_day_feature_rows(counts: pd.DataFrame, valid_positive_days: pd.DataFrame) -> pd.DataFrame:
    """Compute per-day shares before averaging to location-level features."""
    keys = ["location_group_id", "date"]
    base = valid_positive_days[keys + ["daily_total", "is_weekend", "month", "day_of_week"]].copy()
    base["is_weekday"] = ~base["is_weekend"].astype(bool)
    base["is_summer"] = base["month"].isin([6, 7, 8])
    base["is_winter"] = base["month"].isin([12, 1, 2])
    eligible_keys = base[keys]
    counts = counts.merge(eligible_keys, on=keys, how="inner")
    weekday = ~counts["is_weekend"].astype(bool)
    weekend = counts["is_weekend"].astype(bool)
    frames = [
        window_daily_shares(counts, base, weekday & counts["hour"].between(7, 8), "weekday_morning_peak_share_day"),
        window_daily_shares(counts, base, weekday & counts["hour"].between(16, 17), "weekday_evening_peak_share_day"),
        window_daily_shares(counts, base, weekday & counts["hour"].between(10, 14), "weekday_midday_share_day"),
        window_daily_shares(counts, base, weekend & counts["hour"].between(11, 16), "weekend_midday_afternoon_share_day"),
        window_daily_shares(counts, base, counts["hour"].ge(22) | counts["hour"].lt(5), "night_share_day"),
        rolling_2h_peak_daily_shares(counts, base),
    ]
    output = base
    for frame in frames:
        output = output.merge(frame, on=keys, how="left")
    return output


def safe_log_ratio(numerator_mean: float, denominator_mean: float) -> float:
    """Compute log((num + 1) / (den + 1)) safely."""
    if pd.isna(numerator_mean) or pd.isna(denominator_mean):
        return np.nan
    return float(math.log((float(numerator_mean) + 1.0) / (float(denominator_mean) + 1.0)))


def mean_or_nan(values: pd.Series) -> float:
    """Return mean or NaN for empty/non-finite series."""
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def cv_or_nan(values: pd.Series) -> float:
    """Return coefficient of variation or NaN when mean is unavailable/zero."""
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    mean = float(clean.mean())
    if abs(mean) <= 1e-12:
        return np.nan
    return float(clean.std(ddof=0)) / mean


def build_location_features(
    day_features: pd.DataFrame,
    valid_positive_days: pd.DataFrame,
    eligible: pd.DataFrame,
    location_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate per-day features to one row per eligible location group."""
    eligible_groups = eligible.loc[eligible["eligible_location_group"], "location_group_id"].astype(str).sort_values().tolist()
    diagnostics = eligible.set_index("location_group_id")
    group_meta = location_groups.set_index("location_group_id") if not location_groups.empty else pd.DataFrame()
    records: list[dict[str, object]] = []
    for location_group_id in eligible_groups:
        days = valid_positive_days.loc[valid_positive_days["location_group_id"] == location_group_id]
        features = day_features.loc[day_features["location_group_id"] == location_group_id]
        weekdays = days.loc[~days["is_weekend"]]
        weekends = days.loc[days["is_weekend"]]
        summer = days.loc[days["month"].isin([6, 7, 8])]
        winter = days.loc[days["month"].isin([12, 1, 2])]
        weekend_mean = mean_or_nan(weekends["daily_total"])
        weekday_mean = mean_or_nan(weekdays["daily_total"])
        summer_mean = mean_or_nan(summer["daily_total"])
        winter_mean = mean_or_nan(winter["daily_total"])
        diag = diagnostics.loc[location_group_id]
        record = {
            "location_group_id": location_group_id,
            "log_weekend_weekday_ratio": safe_log_ratio(weekend_mean, weekday_mean),
            "weekday_morning_peak_share": mean_or_nan(features["weekday_morning_peak_share_day"]),
            "weekday_evening_peak_share": mean_or_nan(features["weekday_evening_peak_share_day"]),
            "weekday_midday_share": mean_or_nan(features["weekday_midday_share_day"]),
            "weekend_midday_afternoon_share": mean_or_nan(features["weekend_midday_afternoon_share_day"]),
            "weekday_peak_concentration_2h": mean_or_nan(features["weekday_peak_concentration_2h_day"]),
            "log_summer_winter_ratio": safe_log_ratio(summer_mean, winter_mean),
            "night_share": mean_or_nan(features["night_share_day"]),
            "daily_variability_cv": cv_or_nan(days["daily_total"]),
            "n_feature_days": int(len(days)),
            "n_feature_weekdays": int(len(weekdays)),
            "n_feature_weekend_days": int(len(weekends)),
            "n_morning_peak_days_used": int(features["weekday_morning_peak_share_day"].notna().sum()),
            "n_evening_peak_days_used": int(features["weekday_evening_peak_share_day"].notna().sum()),
            "n_weekday_midday_days_used": int(features["weekday_midday_share_day"].notna().sum()),
            "n_weekend_midday_afternoon_days_used": int(features["weekend_midday_afternoon_share_day"].notna().sum()),
            "n_weekday_peak_concentration_days_used": int(features["weekday_peak_concentration_2h_day"].notna().sum()),
            "n_night_share_days_used": int(features["night_share_day"].notna().sum()),
            "n_valid_days": int(diag.get("n_valid_days", 0)),
            "n_valid_weekdays": int(diag.get("n_valid_weekdays", 0)),
            "n_valid_weekend_days": int(diag.get("n_valid_weekend_days", 0)),
            "n_valid_summer_days": int(diag.get("n_valid_summer_days", 0)),
            "n_valid_winter_days": int(diag.get("n_valid_winter_days", 0)),
            "n_months_with_valid_data": int(diag.get("n_months_with_valid_data", 0)),
            "n_years_with_valid_data": int(diag.get("n_years_with_valid_data", 0)),
            "mean_daily_total": float(diag.get("mean_daily_total", np.nan)),
            "median_daily_total": float(diag.get("median_daily_total", np.nan)),
            "percentage_full_coverage_intervals": float(diag.get("percentage_full_coverage_intervals", np.nan)),
            "percentage_partial_coverage_intervals": float(diag.get("percentage_partial_coverage_intervals", np.nan)),
        }
        if not group_meta.empty and location_group_id in group_meta.index:
            record["n_sites_in_group"] = int(group_meta.loc[location_group_id].get("n_sites_in_group", 0))
        else:
            record["n_sites_in_group"] = np.nan
        records.append(record)
    output = pd.DataFrame.from_records(records)
    output["selected_kmeans_features"] = ",".join(SELECTED_KMEANS_FEATURES)
    output["optional_candidate_features"] = ",".join(OPTIONAL_CANDIDATE_FEATURES)
    return output


def assert_valid_kmeans_features(feature_columns: list[str], feature_table: pd.DataFrame) -> None:
    """Prevent forbidden columns or missing/non-numeric columns from entering KMeans matrix."""
    forbidden = sorted(set(feature_columns) & FORBIDDEN_CLUSTERING_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden columns selected for KMeans: {', '.join(forbidden)}")
    missing = sorted(set(feature_columns) - set(feature_table.columns))
    if missing:
        raise AssertionError(f"Selected KMeans features missing from feature table: {', '.join(missing)}")
    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(feature_table[column])]
    if non_numeric:
        raise AssertionError(f"Selected KMeans features must be numeric: {', '.join(non_numeric)}")


def build_report(
    feature_table: pd.DataFrame,
    eligible: pd.DataFrame,
    day_features: pd.DataFrame,
) -> dict[str, object]:
    """Build feature-engineering report."""
    missing_by_feature = {column: int(feature_table[column].isna().sum()) for column in SELECTED_KMEANS_FEATURES + OPTIONAL_CANDIDATE_FEATURES}
    ranges = {}
    for column in SELECTED_KMEANS_FEATURES + OPTIONAL_CANDIDATE_FEATURES:
        values = pd.to_numeric(feature_table[column], errors="coerce")
        ranges[column] = {
            "min": float(values.min()) if values.notna().any() else None,
            "median": float(values.median()) if values.notna().any() else None,
            "mean": float(values.mean()) if values.notna().any() else None,
            "max": float(values.max()) if values.notna().any() else None,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "counts": str(DEFAULT_COUNTS_INPUT.relative_to(PROJECT_ROOT)),
            "day_quality": str(DEFAULT_DAY_QUALITY_INPUT.relative_to(PROJECT_ROOT)),
            "eligible_location_groups": str(DEFAULT_ELIGIBLE_INPUT.relative_to(PROJECT_ROOT)),
        },
        "n_eligible_location_groups_input": int(eligible["eligible_location_group"].sum()),
        "n_location_groups_with_features": int(len(feature_table)),
        "n_day_feature_rows": int(len(day_features)),
        "window_coverage_threshold": WINDOW_COVERAGE_THRESHOLD,
        "selected_kmeans_features": SELECTED_KMEANS_FEATURES,
        "optional_candidate_features_not_selected_by_default": OPTIONAL_CANDIDATE_FEATURES,
        "forbidden_clustering_features": sorted(FORBIDDEN_CLUSTERING_FEATURES),
        "missing_values_by_feature": missing_by_feature,
        "feature_ranges": ranges,
        "methodological_notes": [
            "Only eligible location groups are used.",
            "Valid days only are used, and days with daily_total <= 0 are excluded from share calculations.",
            "Shares are computed per day first and then averaged across days so high-volume days do not dominate.",
            "Absolute volume, quality, geography, and identifiers are diagnostics only and must not enter the KMeans matrix.",
            "night_share and daily_variability_cv are candidate diagnostics and are not selected by default for clustering.",
        ],
        "outputs": {
            "location_group_features_candidate": str(DEFAULT_FEATURE_OUTPUT.relative_to(PROJECT_ROOT)),
            "feature_engineering_report": str(DEFAULT_REPORT_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }


def run_feature_engineering(
    counts_input: Path = DEFAULT_COUNTS_INPUT,
    day_quality_input: Path = DEFAULT_DAY_QUALITY_INPUT,
    eligible_input: Path = DEFAULT_ELIGIBLE_INPUT,
    location_groups_input: Path = DEFAULT_LOCATION_GROUPS_INPUT,
    feature_output: Path = DEFAULT_FEATURE_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, object]:
    """Run location-group feature engineering and write outputs."""
    counts, day_quality, eligible, location_groups = load_inputs(
        counts_input,
        day_quality_input,
        eligible_input,
        location_groups_input,
    )
    eligible_ids = set(eligible.loc[eligible["eligible_location_group"], "location_group_id"].astype(str))
    valid_positive_days = day_quality.loc[
        day_quality["location_group_id"].astype(str).isin(eligible_ids)
        & day_quality["valid_day"]
        & (day_quality["daily_total"] > 0)
    ].copy()
    counts = counts.loc[counts["location_group_id"].astype(str).isin(eligible_ids)].copy()
    day_features = build_day_feature_rows(counts, valid_positive_days)
    feature_table = build_location_features(day_features, valid_positive_days, eligible, location_groups)
    assert_valid_kmeans_features(SELECTED_KMEANS_FEATURES, feature_table)
    report = build_report(feature_table, eligible, day_features)

    feature_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(feature_output, index=False)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineer location-group temporal features for KMeans clustering.")
    parser.add_argument("--counts-input", type=Path, default=DEFAULT_COUNTS_INPUT)
    parser.add_argument("--day-quality-input", type=Path, default=DEFAULT_DAY_QUALITY_INPUT)
    parser.add_argument("--eligible-input", type=Path, default=DEFAULT_ELIGIBLE_INPUT)
    parser.add_argument("--location-groups-input", type=Path, default=DEFAULT_LOCATION_GROUPS_INPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_feature_engineering(
            counts_input=args.counts_input,
            day_quality_input=args.day_quality_input,
            eligible_input=args.eligible_input,
            location_groups_input=args.location_groups_input,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Location-group feature engineering complete")
    print(f"- eligible groups input: {report['n_eligible_location_groups_input']}")
    print(f"- groups with features: {report['n_location_groups_with_features']}")
    print(f"- day feature rows: {report['n_day_feature_rows']}")
    print(f"- selected KMeans features: {report['selected_kmeans_features']}")
    print(f"- optional candidate features: {report['optional_candidate_features_not_selected_by_default']}")
    print(f"- missing values by feature: {report['missing_values_by_feature']}")
    print(f"- outputs: {report['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
