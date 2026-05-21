"""Create final audit table for the small strongly seasonal cluster."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_INPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
QUALITY_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_quality_summary.csv"
GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
FEATURES_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_candidate.csv"
AUDIT_OUTPUT = PROJECT_ROOT / "outputs" / "final_small_cluster_audit.csv"

COMPACT5_FEATURES = [
    "log_weekend_weekday_ratio",
    "weekday_commute_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "log_summer_winter_ratio",
]

QUALITY_COLUMNS = [
    "n_valid_days",
    "n_valid_weekdays",
    "n_valid_weekend_days",
    "n_valid_summer_days",
    "n_valid_winter_days",
    "n_months_with_valid_data",
    "n_years_with_valid_data",
    "percentage_full_coverage_intervals",
    "percentage_partial_coverage_intervals",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required audit inputs."""
    for path in [CLUSTERS_INPUT, QUALITY_INPUT, GROUPS_INPUT, FEATURES_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    return (
        pd.read_csv(CLUSTERS_INPUT),
        pd.read_csv(QUALITY_INPUT),
        pd.read_csv(GROUPS_INPUT),
        pd.read_csv(FEATURES_INPUT),
    )


def identify_small_cluster(clusters: pd.DataFrame) -> int:
    """Identify the smallest final k=2 cluster."""
    return int(clusters["cluster_id"].value_counts().sort_values().index[0])


def quality_flag(row: pd.Series) -> tuple[str, str]:
    """Assign transparent quality flag based on candidate threshold weaknesses."""
    checks = [
        ("n_valid_days < 365", row["n_valid_days"] < 365),
        ("n_valid_weekdays < 240", row["n_valid_weekdays"] < 240),
        ("n_valid_weekend_days < 90", row["n_valid_weekend_days"] < 90),
        ("n_valid_summer_days < 50", row["n_valid_summer_days"] < 50),
        ("n_valid_winter_days < 50", row["n_valid_winter_days"] < 50),
        ("n_months_with_valid_data < 12", row["n_months_with_valid_data"] < 12),
        ("n_years_with_valid_data < 2", row["n_years_with_valid_data"] < 2),
        ("percentage_full_coverage_intervals < 0.90", row["percentage_full_coverage_intervals"] < 0.90),
        ("percentage_partial_coverage_intervals > 0.10", row["percentage_partial_coverage_intervals"] > 0.10),
    ]
    reasons = [reason for reason, failed in checks if bool(failed)]
    if reasons:
        return "check", "; ".join(reasons)
    return "ok", "quality metrics sufficient under audit thresholds"


def build_audit_table(
    clusters: pd.DataFrame,
    quality: pd.DataFrame,
    groups: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Build final small-cluster audit table."""
    small_cluster_id = identify_small_cluster(clusters)
    small = clusters.loc[clusters["cluster_id"] == small_cluster_id].copy()

    if "weekday_commute_peak_share" not in features.columns:
        features = features.copy()
        features["weekday_commute_peak_share"] = (
            pd.to_numeric(features["weekday_morning_peak_share"], errors="coerce")
            + pd.to_numeric(features["weekday_evening_peak_share"], errors="coerce")
        )

    group_columns = ["location_group_id", "gemeente_values", "site_ids", "n_sites_in_group"]
    group_columns = [column for column in group_columns if column in groups.columns]
    audit = small[["location_group_id", "cluster_id", "cluster_name", "silhouette_value", "distance_to_assigned_centroid"]].merge(
        groups[group_columns],
        on="location_group_id",
        how="left",
    )
    audit = audit.merge(quality[["location_group_id", *QUALITY_COLUMNS]], on="location_group_id", how="left")
    audit = audit.merge(features[["location_group_id", *COMPACT5_FEATURES]], on="location_group_id", how="left")

    flags = audit.apply(quality_flag, axis=1, result_type="expand")
    audit["data_quality_flag"] = flags[0]
    audit["data_quality_reason"] = flags[1]

    rename = {"gemeente_values": "gemeente"}
    audit = audit.rename(columns=rename)
    ordered_columns = [
        "location_group_id",
        "gemeente",
        "site_ids",
        "n_sites_in_group",
        *QUALITY_COLUMNS,
        *COMPACT5_FEATURES,
        "silhouette_value",
        "distance_to_assigned_centroid",
        "data_quality_flag",
        "data_quality_reason",
    ]
    audit = audit[ordered_columns].sort_values("log_summer_winter_ratio", ascending=False)
    return audit, small_cluster_id


def run_audit() -> dict[str, object]:
    """Run the final small-cluster audit."""
    clusters, quality, groups, features = load_inputs()
    audit, small_cluster_id = build_audit_table(clusters, quality, groups, features)
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_OUTPUT, index=False)
    return {
        "small_cluster_id": small_cluster_id,
        "n_rows": int(len(audit)),
        "flag_counts": audit["data_quality_flag"].value_counts().to_dict(),
        "outputs": {
            "audit": str(AUDIT_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final small strongly seasonal cluster.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        result = run_audit()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Final small-cluster audit complete")
    print(f"- small cluster id: {result['small_cluster_id']}")
    print(f"- rows: {result['n_rows']}")
    print(f"- flag counts: {result['flag_counts']}")
    print(f"- outputs: {result['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
