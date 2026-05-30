"""Group nearby AWV counting sites and aggregate counts to location groups.

Sites within 100 meters are treated as one counting location using complete-linkage
clustering in Belgian Lambert 72 / EPSG:31370. Counts are summed across original
sites only after preserving interval coverage diagnostics.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inspect_data import DEFAULT_RAW_DIR, PROJECT_ROOT, detect_schema, read_csv_with_schema, resolve_raw_dir  # noqa: E402

DEFAULT_COUNTS_INPUT = PROJECT_ROOT / "data" / "processed" / "counts_site_level_2023_2025.parquet"
DEFAULT_SITES_INPUT = DEFAULT_RAW_DIR / "sites.csv"
DEFAULT_MAPPING_OUTPUT = PROJECT_ROOT / "data" / "processed" / "site_to_location_group.csv"
DEFAULT_GROUPS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
DEFAULT_COUNTS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "counts_location_group_2023_2025.parquet"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "location_grouping_report.json"
# Nearby counters are treated as the same counting location before feature
# engineering, so profiles are based on locations rather than individual devices.
DISTANCE_THRESHOLD_METERS = 100.0


@dataclass(frozen=True)
class LocationGroupingConfig:
    """Runtime configuration for location grouping."""

    distance_threshold_meters: float = DISTANCE_THRESHOLD_METERS


def load_sites(sites_path: Path) -> pd.DataFrame:
    """Load sites.csv, normalize columns, and keep valid coordinates."""
    sites_path = sites_path.resolve()
    schema = detect_schema(sites_path, sites_path.parent)
    sites = read_csv_with_schema(sites_path, schema)
    required = {"site_id", "long", "lat", "gemeente", "naam", "site_nr"}
    missing = sorted(required - set(sites.columns))
    if missing:
        raise ValueError(f"sites.csv missing required columns after normalization: {', '.join(missing)}")

    sites = sites.copy()
    sites["site_id"] = pd.to_numeric(sites["site_id"], errors="coerce").astype("Int64")
    sites["long"] = pd.to_numeric(sites["long"], errors="coerce")
    sites["lat"] = pd.to_numeric(sites["lat"], errors="coerce")
    valid = sites.dropna(subset=["site_id", "long", "lat"]).copy()
    valid = valid.loc[valid["lat"].between(-90, 90) & valid["long"].between(-180, 180)].copy()
    valid["site_id"] = valid["site_id"].astype(int)
    return valid.sort_values("site_id").reset_index(drop=True)


def project_sites(sites: pd.DataFrame) -> pd.DataFrame:
    """Project longitude/latitude from EPSG:4326 to EPSG:31370."""
    # Distances must be computed in metres, not in raw longitude/latitude degrees.
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
    x, y = transformer.transform(sites["long"].to_numpy(), sites["lat"].to_numpy())
    projected = sites.copy()
    projected["x_31370"] = x
    projected["y_31370"] = y
    return projected


def complete_linkage_groups(projected_sites: pd.DataFrame, threshold_meters: float) -> pd.DataFrame:
    """Assign complete-linkage location groups using projected coordinates."""
    sites = projected_sites.copy()
    if sites.empty:
        raise ValueError("No sites with valid coordinates are available for grouping")
    if len(sites) == 1:
        sites["_cluster"] = 1
    else:
        coordinates = sites[["x_31370", "y_31370"]].to_numpy(dtype="float64")
        condensed = pdist(coordinates, metric="euclidean")
        # Complete linkage avoids chaining: all sites in a group must stay within
        # the maximum distance threshold from one another.
        tree = linkage(condensed, method="complete")
        sites["_cluster"] = fcluster(tree, t=threshold_meters, criterion="distance")

    group_order = (
        sites.groupby("_cluster", as_index=False, observed=True)["site_id"]
        .min()
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    group_order["location_group_id"] = [f"LG{idx:04d}" for idx in range(1, len(group_order) + 1)]
    sites = sites.merge(group_order[["_cluster", "location_group_id"]], on="_cluster", how="left", validate="many_to_one")
    sites["n_sites_in_group"] = sites.groupby("location_group_id", observed=True)["site_id"].transform("size")
    return sites.drop(columns=["_cluster"]).sort_values(["location_group_id", "site_id"]).reset_index(drop=True)


def max_pairwise_distance(group: pd.DataFrame) -> float:
    """Return max pairwise projected distance within a group."""
    if len(group) <= 1:
        return 0.0
    distances = pdist(group[["x_31370", "y_31370"]].to_numpy(dtype="float64"), metric="euclidean")
    return float(distances.max()) if len(distances) else 0.0


def pipe_join(values: pd.Series) -> str:
    """Return sorted unique values joined by pipe."""
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    return "|".join(sorted(cleaned.unique()))


def build_outputs_for_groups(grouped_sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create site mapping and location group metadata tables."""
    mapping = grouped_sites[["site_id", "location_group_id", "n_sites_in_group"]].copy()
    records: list[dict[str, object]] = []
    for location_group_id, group in grouped_sites.groupby("location_group_id", sort=True, observed=True):
        records.append(
            {
                "location_group_id": location_group_id,
                "n_sites_in_group": int(len(group)),
                "site_ids": "|".join(str(value) for value in sorted(group["site_id"].astype(int).tolist())),
                "mean_longitude": float(group["long"].mean()),
                "mean_latitude": float(group["lat"].mean()),
                "gemeente_values": pipe_join(group["gemeente"]),
                "naam_values": pipe_join(group["naam"]),
                "site_nr_values": pipe_join(group["site_nr"]),
                "max_pairwise_distance_m": max_pairwise_distance(group),
            }
        )
    groups = pd.DataFrame.from_records(records).sort_values("location_group_id").reset_index(drop=True)
    return mapping, groups


def load_counts(counts_path: Path) -> pd.DataFrame:
    """Load processed site-level counts needed for location aggregation."""
    if not counts_path.exists():
        raise FileNotFoundError(f"Processed counts parquet not found: {counts_path}")
    columns = [
        "site_id",
        "interval_start",
        "count",
        "interval_status",
        "has_dst_spring_75min",
        "duration_minutes_min",
        "duration_minutes_max",
    ]
    counts = pd.read_parquet(counts_path, columns=columns)
    counts = counts.copy()
    counts["site_id"] = pd.to_numeric(counts["site_id"], errors="coerce").astype("Int64")
    counts["interval_start"] = pd.to_datetime(counts["interval_start"])
    counts["count"] = pd.to_numeric(counts["count"], errors="coerce").fillna(0)
    return counts


def aggregate_location_counts(counts: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join location groups and aggregate interval counts across original sites."""
    counts_with_group = counts.merge(mapping, on="site_id", how="left", validate="many_to_one")
    missing_group_rows = int(counts_with_group["location_group_id"].isna().sum())
    counts_with_group = counts_with_group.dropna(subset=["location_group_id"]).copy()
    counts_with_group["site_present"] = 1
    counts_with_group["has_dst_spring_75min"] = counts_with_group["has_dst_spring_75min"].fillna(False).astype(bool)

    # Counts are summed, but coverage keeps track of whether all expected sites
    # contributed to each interval.
    aggregated = (
        counts_with_group.groupby(["location_group_id", "interval_start"], as_index=False, observed=True)
        .agg(
            count=("count", "sum"),
            n_sites_present=("site_id", "nunique"),
            n_sites_expected=("n_sites_in_group", "first"),
            has_dst_spring_75min=("has_dst_spring_75min", "max"),
            duration_minutes_min=("duration_minutes_min", "min"),
            duration_minutes_max=("duration_minutes_max", "max"),
        )
        .sort_values(["location_group_id", "interval_start"])
        .reset_index(drop=True)
    )
    # Partial coverage is kept as diagnostics instead of filling missing sites
    # with zero counts.
    aggregated["interval_full_coverage"] = aggregated["n_sites_present"] == aggregated["n_sites_expected"]
    aggregated["interval_coverage_ratio"] = aggregated["n_sites_present"] / aggregated["n_sites_expected"]
    aggregated["date"] = aggregated["interval_start"].dt.date.astype("string")
    aggregated["year"] = aggregated["interval_start"].dt.year.astype("Int64")
    aggregated["month"] = aggregated["interval_start"].dt.month.astype("Int64")
    aggregated["day_of_week"] = aggregated["interval_start"].dt.dayofweek.astype("Int64")
    aggregated["is_weekend"] = aggregated["day_of_week"].isin([5, 6])
    aggregated["hour"] = aggregated["interval_start"].dt.hour.astype("Int64")
    aggregated["minute"] = aggregated["interval_start"].dt.minute.astype("Int64")
    aggregated["quarter_index"] = (aggregated["hour"] * 4 + aggregated["minute"] // 15).astype("Int64")
    aggregated["interval_status"] = "normal_15min"
    aggregated.loc[aggregated["has_dst_spring_75min"], "interval_status"] = "dst_spring_75min"
    output_columns = [
        "location_group_id",
        "interval_start",
        "count",
        "n_sites_present",
        "n_sites_expected",
        "interval_full_coverage",
        "interval_coverage_ratio",
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
        "duration_minutes_min",
        "duration_minutes_max",
    ]
    return aggregated[output_columns], missing_group_rows


def distribution(values: pd.Series) -> dict[str, int]:
    """Return value counts as a string-keyed dict."""
    return {str(key): int(value) for key, value in values.value_counts(dropna=False).sort_index().to_dict().items()}


def build_report(
    sites: pd.DataFrame,
    grouped_sites: pd.DataFrame,
    groups: pd.DataFrame,
    counts: pd.DataFrame,
    location_counts: pd.DataFrame,
    missing_group_rows: int,
    threshold_meters: float,
) -> dict[str, object]:
    """Build machine-readable location grouping report."""
    n_groups = int(len(groups))
    max_distance = float(groups["max_pairwise_distance_m"].max()) if not groups.empty else 0.0
    over_threshold = groups.loc[groups["max_pairwise_distance_m"] > threshold_meters]
    warnings: list[str] = []
    if not over_threshold.empty:
        warnings.append(f"{len(over_threshold)} location groups have max pairwise distance above {threshold_meters} meters.")
    if missing_group_rows > 0:
        warnings.append(f"{missing_group_rows} count rows had no valid-coordinate location group and were excluded.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "complete-linkage clustering on EPSG:31370 projected site coordinates",
        "distance_threshold_meters": threshold_meters,
        "number_of_original_sites": int(len(sites)),
        "number_of_count_sites": int(counts["site_id"].nunique()),
        "number_of_count_rows_input": int(len(counts)),
        "number_of_count_rows_without_location_group": missing_group_rows,
        "number_of_location_groups": n_groups,
        "groups_with_one_site": int((groups["n_sites_in_group"] == 1).sum()),
        "groups_with_two_sites": int((groups["n_sites_in_group"] == 2).sum()),
        "groups_with_more_than_two_sites": int((groups["n_sites_in_group"] > 2).sum()),
        "maximum_within_group_distance_m": max_distance,
        "n_sites_in_group_distribution": distribution(groups["n_sites_in_group"]),
        "location_count_rows_output": int(len(location_counts)),
        "location_groups_in_counts": int(location_counts["location_group_id"].nunique()),
        "interval_full_coverage_distribution": distribution(location_counts["interval_full_coverage"]),
        "interval_coverage_ratio_summary": {
            "min": float(location_counts["interval_coverage_ratio"].min()) if not location_counts.empty else None,
            "median": float(location_counts["interval_coverage_ratio"].median()) if not location_counts.empty else None,
            "mean": float(location_counts["interval_coverage_ratio"].mean()) if not location_counts.empty else None,
            "max": float(location_counts["interval_coverage_ratio"].max()) if not location_counts.empty else None,
        },
        "warnings": warnings,
        "methodological_notes": [
            "Counts are summed across original sites within each location group before feature engineering.",
            "Partial coverage is tracked with n_sites_present, n_sites_expected, interval_full_coverage, and interval_coverage_ratio.",
            "Missing intervals are not imputed as zero.",
            "Absolute volume should not be used as a clustering feature later.",
        ],
        "outputs": {
            "site_to_location_group": str(DEFAULT_MAPPING_OUTPUT.relative_to(PROJECT_ROOT)),
            "location_groups": str(DEFAULT_GROUPS_OUTPUT.relative_to(PROJECT_ROOT)),
            "counts_location_group": str(DEFAULT_COUNTS_OUTPUT.relative_to(PROJECT_ROOT)),
            "report": str(DEFAULT_REPORT_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }


def run_location_grouping(
    counts_input: Path = DEFAULT_COUNTS_INPUT,
    sites_input: Path = DEFAULT_SITES_INPUT,
    mapping_output: Path = DEFAULT_MAPPING_OUTPUT,
    groups_output: Path = DEFAULT_GROUPS_OUTPUT,
    counts_output: Path = DEFAULT_COUNTS_OUTPUT,
    report_output: Path = DEFAULT_REPORT_OUTPUT,
    config: LocationGroupingConfig = LocationGroupingConfig(),
) -> dict[str, object]:
    """Run site location grouping and count aggregation."""
    # 1. Read AWV site coordinates. If no explicit path is given, use data/raw.
    sites_input = sites_input.resolve()
    if not sites_input.exists():
        raw_dir = resolve_raw_dir(DEFAULT_RAW_DIR)
        sites_input = raw_dir / "sites.csv"
    sites = load_sites(sites_input)

    # 2. Work in Belgian Lambert coordinates before measuring metres.
    projected = project_sites(sites)

    # 3. Group sites within 100 metres using complete linkage to avoid chaining.
    grouped_sites = complete_linkage_groups(projected, threshold_meters=config.distance_threshold_meters)
    mapping, groups = build_outputs_for_groups(grouped_sites)

    # 4. Sum the already-cleaned site counts to location-group level.
    counts = load_counts(counts_input)
    location_counts, missing_group_rows = aggregate_location_counts(counts, mapping)

    # 5. Save both the mapping and the aggregated time series for auditability.
    report = build_report(
        sites=sites,
        grouped_sites=grouped_sites,
        groups=groups,
        counts=counts,
        location_counts=location_counts,
        missing_group_rows=missing_group_rows,
        threshold_meters=config.distance_threshold_meters,
    )

    for path in [mapping_output, groups_output, counts_output, report_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(mapping_output, index=False)
    groups.to_csv(groups_output, index=False)
    location_counts.to_parquet(counts_output, index=False)
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group nearby AWV sites and aggregate counts to location groups.")
    parser.add_argument("--counts-input", type=Path, default=DEFAULT_COUNTS_INPUT)
    parser.add_argument("--sites-input", type=Path, default=DEFAULT_SITES_INPUT)
    parser.add_argument("--distance-threshold-meters", type=float, default=DISTANCE_THRESHOLD_METERS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_location_grouping(
            counts_input=args.counts_input,
            sites_input=args.sites_input,
            config=LocationGroupingConfig(distance_threshold_meters=args.distance_threshold_meters),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Location grouping complete")
    print(f"- original sites: {report['number_of_original_sites']}")
    print(f"- location groups: {report['number_of_location_groups']}")
    print(f"- one-site groups: {report['groups_with_one_site']}")
    print(f"- two-site groups: {report['groups_with_two_sites']}")
    print(f"- >2-site groups: {report['groups_with_more_than_two_sites']}")
    print(f"- max within-group distance: {report['maximum_within_group_distance_m']:.2f} m")
    print(f"- location count rows: {report['location_count_rows_output']}")
    if report["warnings"]:
        print("- warnings:")
        for warning in report["warnings"]:
            print(f"  {warning}")
    print(f"- outputs: {report['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
