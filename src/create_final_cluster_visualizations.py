"""Create final visualizations for AWV KMeans location-group clusters."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_INPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
COUNTS_INPUT = PROJECT_ROOT / "data" / "processed" / "counts_location_group_2023_2025.parquet"
LOCATION_GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"

FEATURE_PROFILE_FIG = PROJECT_ROOT / "outputs" / "fig_feature_profile_standardized_k2.png"
RAW_FEATURE_TABLE = PROJECT_ROOT / "outputs" / "table_raw_feature_summary_k2.csv"
WEEKDAY_PROFILE_FIG = PROJECT_ROOT / "outputs" / "fig_weekday_hourly_profile_k2.png"
WEEKEND_PROFILE_FIG = PROJECT_ROOT / "outputs" / "fig_weekend_hourly_profile_k2.png"
MONTHLY_PROFILE_FIG = PROJECT_ROOT / "outputs" / "fig_monthly_seasonal_profile_k2.png"
MAP_FIG = PROJECT_ROOT / "outputs" / "fig_location_groups_map_k2.png"
SMALL_CLUSTER_TABLE = PROJECT_ROOT / "outputs" / "small_cluster_detail_k2.csv"

COMPACT5_FEATURES = [
    "log_weekend_weekday_ratio",
    "weekday_commute_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "log_summer_winter_ratio",
]
CLUSTER_COLORS = {
    0: "#4C78A8",
    1: "#F58518",
}


def setup_matplotlib():
    """Import matplotlib with a writable local cache."""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load final clusters, counts, and location-group metadata."""
    for path in [CLUSTERS_INPUT, COUNTS_INPUT, LOCATION_GROUPS_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    clusters = pd.read_csv(CLUSTERS_INPUT)
    counts = pd.read_parquet(COUNTS_INPUT)
    location_groups = pd.read_csv(LOCATION_GROUPS_INPUT)

    missing = sorted({"location_group_id", "cluster_id", *COMPACT5_FEATURES} - set(clusters.columns))
    if missing:
        raise ValueError(f"Final cluster table missing columns: {', '.join(missing)}")
    return clusters, counts, location_groups


def cluster_characterizations(clusters: pd.DataFrame) -> dict[int, str]:
    """Create cautious temporal characterizations from cluster feature means."""
    means = clusters.groupby("cluster_id")[COMPACT5_FEATURES].mean()
    seasonal_cluster = int(means["log_summer_winter_ratio"].idxmax())
    return {
        int(cluster_id): (
            "strongly seasonal / recreational-like pattern"
            if int(cluster_id) == seasonal_cluster
            else "broad mixed / less seasonal pattern"
        )
        for cluster_id in means.index
    }


def raw_feature_summary(clusters: pd.DataFrame) -> pd.DataFrame:
    """Create mean/median raw compact5 summary per cluster."""
    records: list[dict[str, object]] = []
    characterizations = cluster_characterizations(clusters)
    for cluster_id, rows in clusters.groupby("cluster_id", sort=True):
        record: dict[str, object] = {
            "cluster_id": int(cluster_id),
            "cluster_name": f"cluster_{int(cluster_id)}",
            "n_observations": int(len(rows)),
            "pct_observations": float(len(rows) / len(clusters)),
            "cautious_characterization": characterizations[int(cluster_id)],
        }
        for feature in COMPACT5_FEATURES:
            record[f"mean_{feature}"] = float(rows[feature].mean())
            record[f"median_{feature}"] = float(rows[feature].median())
        records.append(record)
    return pd.DataFrame.from_records(records)


def plot_feature_profile(clusters: pd.DataFrame) -> None:
    """Plot mean standardized compact5 feature values by cluster."""
    plt = setup_matplotlib()
    z_features = [f"z_{feature}" for feature in COMPACT5_FEATURES]
    profile = clusters.groupby("cluster_id")[z_features].mean().rename(
        columns={f"z_{feature}": feature for feature in COMPACT5_FEATURES}
    )
    x = np.arange(len(COMPACT5_FEATURES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset_index, (cluster_id, row) in enumerate(profile.iterrows()):
        offset = (offset_index - (len(profile) - 1) / 2) * width
        ax.bar(
            x + offset,
            row[COMPACT5_FEATURES].to_numpy(),
            width=width,
            label=f"cluster_{int(cluster_id)}",
            color=CLUSTER_COLORS.get(int(cluster_id)),
        )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(COMPACT5_FEATURES, rotation=35, ha="right")
    ax.set_ylabel("Mean standardized feature value")
    ax.set_title("Standardized compact5 feature profile by cluster")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    FEATURE_PROFILE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEATURE_PROFILE_FIG, dpi=180)
    plt.close(fig)


def build_valid_days(counts: pd.DataFrame) -> pd.DataFrame:
    """Recreate valid location-group days using 90% site-interval coverage."""
    counts = counts.copy()
    counts["date"] = pd.to_datetime(counts["date"]).dt.date.astype(str)
    daily = counts.groupby(["location_group_id", "date"], as_index=False).agg(
        daily_total=("count", "sum"),
        n_sites_expected=("n_sites_expected", "max"),
        n_site_interval_observations_present=("n_sites_present", "sum"),
        is_weekend=("is_weekend", "max"),
        month=("month", "max"),
    )
    daily["expected_site_interval_observations"] = 96 * daily["n_sites_expected"]
    daily["site_interval_coverage_ratio"] = (
        daily["n_site_interval_observations_present"] / daily["expected_site_interval_observations"]
    )
    daily["valid_day"] = daily["site_interval_coverage_ratio"] >= 0.90
    return daily.loc[daily["valid_day"] & (daily["daily_total"] > 0)].copy()


def location_level_hourly_profiles(
    counts: pd.DataFrame,
    clusters: pd.DataFrame,
    valid_days: pd.DataFrame,
    weekend: bool,
) -> pd.DataFrame:
    """Compute location-level hourly shares, then average profiles by cluster."""
    valid = valid_days.loc[valid_days["is_weekend"].astype(bool) == weekend, [
        "location_group_id",
        "date",
        "daily_total",
    ]].copy()
    subset = counts.copy()
    subset["date"] = pd.to_datetime(subset["date"]).dt.date.astype(str)
    subset = subset.merge(valid, on=["location_group_id", "date"], how="inner")
    hourly = subset.groupby(["location_group_id", "date", "hour"], as_index=False).agg(
        hour_count=("count", "sum"),
        daily_total=("daily_total", "max"),
    )
    hourly["hourly_share"] = hourly["hour_count"] / hourly["daily_total"]
    location_hour = hourly.groupby(["location_group_id", "hour"], as_index=False).agg(
        mean_day_share=("hourly_share", "mean"),
        n_days=("date", "nunique"),
    )
    location_hour = location_hour.merge(
        clusters[["location_group_id", "cluster_id"]],
        on="location_group_id",
        how="inner",
    )
    profile = location_hour.groupby(["cluster_id", "hour"], as_index=False).agg(
        mean_hourly_share=("mean_day_share", "mean"),
        n_location_groups=("location_group_id", "nunique"),
    )
    all_hours = pd.MultiIndex.from_product(
        [sorted(clusters["cluster_id"].unique()), range(24)],
        names=["cluster_id", "hour"],
    )
    return profile.set_index(["cluster_id", "hour"]).reindex(all_hours).reset_index()


def plot_hourly_profile(profile: pd.DataFrame, title: str, output: Path) -> None:
    """Plot hourly cluster profiles."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster_id, rows in profile.groupby("cluster_id", sort=True):
        ax.plot(
            rows["hour"],
            rows["mean_hourly_share"],
            marker="o",
            linewidth=2,
            label=f"cluster_{int(cluster_id)}",
            color=CLUSTER_COLORS.get(int(cluster_id)),
        )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average hourly share of daily count")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def monthly_profile(
    valid_days: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """Compute normalized monthly seasonal profile by cluster."""
    location_month = valid_days.groupby(["location_group_id", "month"], as_index=False).agg(
        mean_daily_total_month=("daily_total", "mean"),
        n_valid_days=("date", "nunique"),
    )
    location_overall = valid_days.groupby("location_group_id", as_index=False).agg(
        overall_mean_daily_total=("daily_total", "mean"),
    )
    location_month = location_month.merge(location_overall, on="location_group_id", how="left")
    location_month["normalized_monthly_profile"] = (
        location_month["mean_daily_total_month"] / location_month["overall_mean_daily_total"]
    )
    location_month = location_month.merge(clusters[["location_group_id", "cluster_id"]], on="location_group_id", how="inner")
    profile = location_month.groupby(["cluster_id", "month"], as_index=False).agg(
        mean_normalized_monthly_profile=("normalized_monthly_profile", "mean"),
        n_location_groups=("location_group_id", "nunique"),
    )
    all_months = pd.MultiIndex.from_product(
        [sorted(clusters["cluster_id"].unique()), range(1, 13)],
        names=["cluster_id", "month"],
    )
    return profile.set_index(["cluster_id", "month"]).reindex(all_months).reset_index()


def plot_monthly_profile(profile: pd.DataFrame) -> None:
    """Plot normalized monthly seasonal profile."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    for cluster_id, rows in profile.groupby("cluster_id", sort=True):
        ax.plot(
            rows["month"],
            rows["mean_normalized_monthly_profile"],
            marker="o",
            linewidth=2,
            label=f"cluster_{int(cluster_id)}",
            color=CLUSTER_COLORS.get(int(cluster_id)),
        )
    ax.axhline(1, color="#333333", linewidth=1, linestyle="--")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean daily count relative to location mean")
    ax.set_title("Normalized monthly seasonal profile by cluster")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    MONTHLY_PROFILE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MONTHLY_PROFILE_FIG, dpi=180)
    plt.close(fig)


def plot_map(clusters: pd.DataFrame, location_groups: pd.DataFrame) -> None:
    """Plot location groups by cluster using coordinates for interpretation."""
    plt = setup_matplotlib()
    if "mean_longitude" not in clusters.columns or "mean_latitude" not in clusters.columns:
        clusters = clusters.merge(
            location_groups[["location_group_id", "mean_longitude", "mean_latitude"]],
            on="location_group_id",
            how="left",
        )
    plot_data = clusters.dropna(subset=["mean_longitude", "mean_latitude"]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 8))
    for cluster_id, rows in plot_data.groupby("cluster_id", sort=True):
        ax.scatter(
            rows["mean_longitude"],
            rows["mean_latitude"],
            s=46 if len(rows) > 10 else 70,
            alpha=0.82,
            color=CLUSTER_COLORS.get(int(cluster_id)),
            edgecolor="white",
            linewidth=0.6,
            label=f"cluster_{int(cluster_id)}",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("AWV location groups by final KMeans cluster")
    ax.text(
        0.01,
        0.01,
        "Coordinates shown for interpretation only; not used as clustering features.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    MAP_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAP_FIG, dpi=180)
    plt.close(fig)


def small_cluster_detail(clusters: pd.DataFrame) -> pd.DataFrame:
    """Create small-cluster detail table."""
    small_cluster_id = int(clusters["cluster_id"].value_counts().sort_values().index[0])
    columns = [
        "location_group_id",
        "gemeente_values",
        "site_ids",
        *COMPACT5_FEATURES,
        "silhouette_value",
        "distance_to_assigned_centroid",
    ]
    available = [column for column in columns if column in clusters.columns]
    return clusters.loc[clusters["cluster_id"] == small_cluster_id, available].sort_values(
        ["log_summer_winter_ratio", "log_weekend_weekday_ratio"],
        ascending=[False, False],
    )


def run_visualizations() -> dict[str, object]:
    """Create all final visualizations and tables."""
    clusters, counts, location_groups = load_inputs()
    plot_feature_profile(clusters)
    summary = raw_feature_summary(clusters)
    RAW_FEATURE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RAW_FEATURE_TABLE, index=False)

    valid_days = build_valid_days(counts)
    weekday = location_level_hourly_profiles(counts, clusters, valid_days, weekend=False)
    weekend = location_level_hourly_profiles(counts, clusters, valid_days, weekend=True)
    plot_hourly_profile(weekday, "Weekday hourly profile by cluster", WEEKDAY_PROFILE_FIG)
    plot_hourly_profile(weekend, "Weekend hourly profile by cluster", WEEKEND_PROFILE_FIG)

    monthly = monthly_profile(valid_days, clusters)
    plot_monthly_profile(monthly)
    plot_map(clusters, location_groups)

    small = small_cluster_detail(clusters)
    SMALL_CLUSTER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    small.to_csv(SMALL_CLUSTER_TABLE, index=False)

    return {
        "n_location_groups": int(len(clusters)),
        "valid_days": int(len(valid_days)),
        "cluster_sizes": clusters["cluster_id"].value_counts().sort_index().astype(int).to_dict(),
        "outputs": {
            "feature_profile": str(FEATURE_PROFILE_FIG.relative_to(PROJECT_ROOT)),
            "raw_feature_table": str(RAW_FEATURE_TABLE.relative_to(PROJECT_ROOT)),
            "weekday_hourly_profile": str(WEEKDAY_PROFILE_FIG.relative_to(PROJECT_ROOT)),
            "weekend_hourly_profile": str(WEEKEND_PROFILE_FIG.relative_to(PROJECT_ROOT)),
            "monthly_profile": str(MONTHLY_PROFILE_FIG.relative_to(PROJECT_ROOT)),
            "map": str(MAP_FIG.relative_to(PROJECT_ROOT)),
            "small_cluster_table": str(SMALL_CLUSTER_TABLE.relative_to(PROJECT_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final KMeans cluster visualizations.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        result = run_visualizations()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Final cluster visualizations complete")
    print(f"- location groups: {result['n_location_groups']}")
    print(f"- valid positive days used for temporal profiles: {result['valid_days']}")
    print(f"- cluster sizes: {result['cluster_sizes']}")
    print(f"- outputs: {result['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
