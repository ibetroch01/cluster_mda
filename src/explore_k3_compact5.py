"""Explore global k=3 KMeans compared with the final k=2 compact5 model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from plot_style import (
    CLUSTER_COLORS,
    FEATURE_LABELS,
    K3_CLUSTER_LABELS,
    MONTH_LABELS,
    clean_axes,
    save_figure,
    setup_matplotlib,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_candidate.csv"
K2_INPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
LOCATION_GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
COUNTS_INPUT = PROJECT_ROOT / "data" / "processed" / "counts_location_group_2023_2025.parquet"

K3_CLUSTERS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "location_group_clusters_k3_compact5.csv"
TRANSITION_OUTPUT = PROJECT_ROOT / "outputs" / "k2_to_k3_transition_table.csv"
K3_SUMMARY_OUTPUT = PROJECT_ROOT / "outputs" / "k3_compact5_cluster_summary.csv"
BROAD_SPLIT_OUTPUT = PROJECT_ROOT / "outputs" / "k3_broad_cluster_split_summary.csv"
BROAD_DIFF_OUTPUT = PROJECT_ROOT / "outputs" / "k3_broad_cluster_feature_differences.csv"
FIG_K3_PROFILE = PROJECT_ROOT / "outputs" / "fig_k3_standardized_feature_profile.png"
FIG_BROAD_PROFILE = PROJECT_ROOT / "outputs" / "fig_k3_broad_split_feature_profile.png"
FIG_WEEKDAY = PROJECT_ROOT / "outputs" / "fig_k3_weekday_hourly_profile.png"
FIG_WEEKEND = PROJECT_ROOT / "outputs" / "fig_k3_weekend_hourly_profile.png"
FIG_MONTHLY = PROJECT_ROOT / "outputs" / "fig_k3_monthly_profile.png"
FIG_HEATMAP = PROJECT_ROOT / "outputs" / "fig_k2_to_k3_transition_heatmap.png"

COMPACT5_FEATURES = [
    "log_weekend_weekday_ratio",
    "weekday_commute_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "log_summer_winter_ratio",
]
KMEANS_PARAMETERS = {
    # k=3 is exploratory only; the final model remains compact5 with k=2.
    "n_clusters": 3,
    "init": "k-means++",
    "n_init": 100,
    "max_iter": 500,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "random_state": 42,
}
COLORS = {
    0: CLUSTER_COLORS["mixed"],
    1: CLUSTER_COLORS["seasonal"],
    2: CLUSTER_COLORS["commuter"],
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load feature, final k=2, and location metadata inputs."""
    for path in [FEATURE_INPUT, K2_INPUT, LOCATION_GROUPS_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    features = pd.read_csv(FEATURE_INPUT)
    k2 = pd.read_csv(K2_INPUT)
    groups = pd.read_csv(LOCATION_GROUPS_INPUT)
    return features, k2, groups


def create_compact5(features: pd.DataFrame) -> pd.DataFrame:
    """Recreate compact5 exactly as in the final k=2 model."""
    required = {
        "location_group_id",
        "log_weekend_weekday_ratio",
        "weekday_morning_peak_share",
        "weekday_evening_peak_share",
        "weekday_midday_share",
        "weekend_midday_afternoon_share",
        "log_summer_winter_ratio",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature input missing compact5 source columns: {', '.join(missing)}")
    output = features.copy()
    source_columns = sorted(required - {"location_group_id"})
    for column in source_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["weekday_commute_peak_share"] = (
        output["weekday_morning_peak_share"] + output["weekday_evening_peak_share"]
    )
    # Use the same complete-case compact5 matrix as the final k=2 model so the
    # comparison is fair.
    output[COMPACT5_FEATURES] = output[COMPACT5_FEATURES].replace([np.inf, -np.inf], np.nan)
    return output.dropna(subset=COMPACT5_FEATURES).copy()


def fit_k3(compact: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """Fit global k=3 KMeans on standardized compact5 features."""
    matrix = compact[COMPACT5_FEATURES].apply(pd.to_numeric, errors="coerce")
    # Standardization is repeated here because this script fits a separate
    # exploratory model.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(**KMEANS_PARAMETERS)
    labels = model.fit_predict(scaled)
    silhouettes = silhouette_samples(scaled, labels)
    distances = np.linalg.norm(scaled - model.cluster_centers_[labels], axis=1)

    assignments = compact[["location_group_id"]].copy()
    assignments["k3_cluster_id"] = labels.astype(int)
    assignments["k3_cluster_name"] = assignments["k3_cluster_id"].map(lambda value: f"cluster_{value}")
    assignments["k3_silhouette_value"] = silhouettes
    assignments["k3_distance_to_centroid"] = distances
    for index, feature in enumerate(COMPACT5_FEATURES):
        assignments[feature] = matrix[feature].to_numpy()
        assignments[f"z_{feature}"] = scaled[:, index]

    metadata_columns = [
        "location_group_id",
        "n_sites_in_group",
        "site_ids",
        "mean_longitude",
        "mean_latitude",
        "gemeente_values",
        "naam_values",
        "site_nr_values",
        "max_pairwise_distance_m",
    ]
    available = [column for column in metadata_columns if column in groups.columns]
    assignments = assignments.merge(groups[available], on="location_group_id", how="left")
    return assignments, pd.DataFrame(scaled, columns=[f"z_{feature}" for feature in COMPACT5_FEATURES]), model


def build_k3_summary(assignments: pd.DataFrame, overall_silhouette: float, inertia: float) -> pd.DataFrame:
    """Build global k=3 cluster summary."""
    records: list[dict[str, object]] = []
    n_total = len(assignments)
    for cluster_id, rows in assignments.groupby("k3_cluster_id", sort=True):
        record: dict[str, object] = {
            "k3_cluster_id": int(cluster_id),
            "k3_cluster_name": f"cluster_{int(cluster_id)}",
            "n_observations": int(len(rows)),
            "pct_observations": float(len(rows) / n_total),
            "inertia": float(inertia),
            "overall_silhouette": float(overall_silhouette),
            "silhouette_mean": float(rows["k3_silhouette_value"].mean()),
            "silhouette_median": float(rows["k3_silhouette_value"].median()),
            "pct_negative_silhouette": float((rows["k3_silhouette_value"] < 0).mean()),
        }
        z_means = pd.Series({feature: float(rows[f"z_{feature}"].mean()) for feature in COMPACT5_FEATURES})
        top = z_means.reindex(z_means.abs().sort_values(ascending=False).index).head(5)
        record["top_5_signed_features"] = "; ".join(
            f"{'high' if value >= 0 else 'low'} {feature} ({value:.2f} SD)"
            for feature, value in top.items()
        )
        for feature in COMPACT5_FEATURES:
            record[f"raw_mean_{feature}"] = float(rows[feature].mean())
            record[f"raw_median_{feature}"] = float(rows[feature].median())
            record[f"z_mean_{feature}"] = float(rows[f"z_{feature}"].mean())
        records.append(record)
    return pd.DataFrame.from_records(records)


def compare_k2_k3(k2: pd.DataFrame, k3: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Build transition table and identify stable small-cluster overlap."""
    # The transition table shows whether k=3 preserves the small k=2 cluster
    # and how it splits the broad k=2 cluster.
    merged = k3.merge(
        k2[["location_group_id", "cluster_id", "cluster_name"]],
        on="location_group_id",
        how="inner",
    ).rename(columns={"cluster_id": "k2_cluster_id", "cluster_name": "k2_cluster_name"})
    table = pd.crosstab(merged["k2_cluster_id"], merged["k3_cluster_id"])
    table = table.reindex(index=sorted(merged["k2_cluster_id"].unique()), columns=sorted(merged["k3_cluster_id"].unique()), fill_value=0)
    long_records = []
    for k2_cluster_id in table.index:
        row_total = int(table.loc[k2_cluster_id].sum())
        for k3_cluster_id in table.columns:
            count = int(table.loc[k2_cluster_id, k3_cluster_id])
            long_records.append(
                {
                    "k2_cluster_id": int(k2_cluster_id),
                    "k3_cluster_id": int(k3_cluster_id),
                    "n_location_groups": count,
                    "pct_of_k2_cluster": count / row_total if row_total else 0,
                }
            )
    transition_long = pd.DataFrame.from_records(long_records)

    k2_small_id = int(merged["k2_cluster_id"].value_counts().sort_values().index[0])
    k2_broad_id = int(merged["k2_cluster_id"].value_counts().sort_values().index[-1])
    small_members = set(merged.loc[merged["k2_cluster_id"] == k2_small_id, "location_group_id"])
    overlap_counts = (
        merged.loc[merged["k2_cluster_id"] == k2_small_id]
        .groupby("k3_cluster_id")["location_group_id"]
        .count()
        .sort_values(ascending=False)
    )
    corresponding_k3 = int(overlap_counts.index[0])
    k3_members = set(merged.loc[merged["k3_cluster_id"] == corresponding_k3, "location_group_id"])
    intersection = small_members & k3_members
    union = small_members | k3_members
    broad_split_clusters = sorted(
        merged.loc[merged["k2_cluster_id"] == k2_broad_id, "k3_cluster_id"].unique().tolist()
    )
    info = {
        "k2_small_cluster_id": k2_small_id,
        "k2_broad_cluster_id": k2_broad_id,
        "corresponding_k3_cluster_id": corresponding_k3,
        "small_cluster_intersection_n": len(intersection),
        "small_cluster_union_n": len(union),
        "small_cluster_jaccard": len(intersection) / len(union) if union else 0,
        "small_cluster_overlap_location_group_ids": sorted(intersection),
        "k3_clusters_splitting_k2_broad": broad_split_clusters,
    }
    return transition_long, info, merged


def build_broad_split_summary(merged: pd.DataFrame, broad_k2_id: int, broad_k3_clusters: list[int]) -> pd.DataFrame:
    """Summarize k=3 subclusters inside the large k=2 cluster."""
    broad = merged.loc[
        (merged["k2_cluster_id"] == broad_k2_id)
        & (merged["k3_cluster_id"].isin(broad_k3_clusters))
    ].copy()
    n_broad = len(merged.loc[merged["k2_cluster_id"] == broad_k2_id])
    records = []
    for cluster_id, rows in broad.groupby("k3_cluster_id", sort=True):
        record: dict[str, object] = {
            "k3_cluster_id": int(cluster_id),
            "k3_cluster_name": f"cluster_{int(cluster_id)}",
            "n_location_groups": int(len(rows)),
            "pct_of_original_k2_broad_cluster": float(len(rows) / n_broad),
            "silhouette_mean": float(rows["k3_silhouette_value"].mean()),
            "silhouette_median": float(rows["k3_silhouette_value"].median()),
            "pct_negative_silhouette": float((rows["k3_silhouette_value"] < 0).mean()),
        }
        for feature in COMPACT5_FEATURES:
            record[f"raw_mean_{feature}"] = float(rows[feature].mean())
            record[f"raw_median_{feature}"] = float(rows[feature].median())
            record[f"z_mean_{feature}"] = float(rows[f"z_{feature}"].mean())
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_broad_feature_differences(broad_summary: pd.DataFrame, merged: pd.DataFrame, broad_k2_id: int) -> pd.DataFrame:
    """Compute direct feature differences between two broad-cluster subclusters."""
    if len(broad_summary) != 2:
        return pd.DataFrame()
    # Differences are reported in raw and standardized units for interpretation.
    cluster_a, cluster_b = broad_summary["k3_cluster_id"].tolist()
    rows_a = merged.loc[(merged["k2_cluster_id"] == broad_k2_id) & (merged["k3_cluster_id"] == cluster_a)]
    rows_b = merged.loc[(merged["k2_cluster_id"] == broad_k2_id) & (merged["k3_cluster_id"] == cluster_b)]
    records = []
    for feature in COMPACT5_FEATURES:
        raw_a = rows_a[feature]
        raw_b = rows_b[feature]
        mean_a = float(raw_a.mean())
        mean_b = float(raw_b.mean())
        z_a = float(rows_a[f"z_{feature}"].mean())
        z_b = float(rows_b[f"z_{feature}"].mean())
        pooled = np.sqrt((raw_a.var(ddof=1) + raw_b.var(ddof=1)) / 2)
        effect = (mean_a - mean_b) / pooled if pooled and not np.isnan(pooled) else np.nan
        records.append(
            {
                "feature": feature,
                "cluster_a": int(cluster_a),
                "cluster_b": int(cluster_b),
                "raw_mean_cluster_a": mean_a,
                "raw_mean_cluster_b": mean_b,
                "raw_mean_difference_a_minus_b": mean_a - mean_b,
                "z_centroid_cluster_a": z_a,
                "z_centroid_cluster_b": z_b,
                "z_centroid_difference_a_minus_b": z_a - z_b,
                "effect_size_raw_cohens_d_a_minus_b": float(effect) if not np.isnan(effect) else np.nan,
                "abs_z_difference": abs(z_a - z_b),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("abs_z_difference", ascending=False)


def plot_standardized_profile(summary: pd.DataFrame, output: Path, title: str) -> None:
    """Plot standardized feature means by cluster."""
    plt, _ = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    y = np.arange(len(COMPACT5_FEATURES))
    height = min(0.24, 0.75 / len(summary))
    for idx, row in enumerate(summary.sort_values("k3_cluster_id").itertuples(index=False)):
        offset = (idx - (len(summary) - 1) / 2) * height
        values = [getattr(row, f"z_mean_{feature}") for feature in COMPACT5_FEATURES]
        cluster_id = int(row.k3_cluster_id)
        ax.barh(
            y + offset,
            values,
            height=height,
            color=COLORS.get(cluster_id),
            label=f"{K3_CLUSTER_LABELS.get(cluster_id, f'cluster_{cluster_id}')} (n={int(getattr(row, 'n_observations', getattr(row, 'n_location_groups', 0)))})",
            alpha=0.95,
        )
    ax.axvline(0, color="#333333", linewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABELS[feature].replace("\n", " ") for feature in COMPACT5_FEATURES])
    ax.set_xlabel("Mean standardized feature value (z-score)")
    ax.set_title(title)
    clean_axes(ax, grid_axis="x")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=1)
    fig.subplots_adjust(bottom=0.30, left=0.28)
    save_figure(fig, output)
    plt.close(fig)


def build_valid_days(counts: pd.DataFrame) -> pd.DataFrame:
    """Recreate valid positive days using 90% site-interval coverage."""
    counts = counts.copy()
    counts["date"] = pd.to_datetime(counts["date"]).dt.date.astype(str)
    daily = counts.groupby(["location_group_id", "date"], as_index=False).agg(
        daily_total=("count", "sum"),
        n_sites_expected=("n_sites_expected", "max"),
        n_site_interval_observations_present=("n_sites_present", "sum"),
        is_weekend=("is_weekend", "max"),
        month=("month", "max"),
    )
    daily["site_interval_coverage_ratio"] = daily["n_site_interval_observations_present"] / (96 * daily["n_sites_expected"])
    return daily.loc[(daily["site_interval_coverage_ratio"] >= 0.90) & (daily["daily_total"] > 0)].copy()


def location_level_hourly_profiles(counts: pd.DataFrame, assignments: pd.DataFrame, valid_days: pd.DataFrame, weekend: bool) -> pd.DataFrame:
    """Compute location-level hourly shares, then average by k=3 cluster."""
    valid = valid_days.loc[valid_days["is_weekend"].astype(bool) == weekend, ["location_group_id", "date", "daily_total"]]
    counts = counts.copy()
    counts["date"] = pd.to_datetime(counts["date"]).dt.date.astype(str)
    subset = counts.merge(valid, on=["location_group_id", "date"], how="inner")
    hourly = subset.groupby(["location_group_id", "date", "hour"], as_index=False).agg(
        hour_count=("count", "sum"),
        daily_total=("daily_total", "max"),
    )
    hourly["hourly_share"] = hourly["hour_count"] / hourly["daily_total"]
    location_hour = hourly.groupby(["location_group_id", "hour"], as_index=False).agg(
        mean_day_share=("hourly_share", "mean"),
        n_days=("date", "nunique"),
    )
    location_hour = location_hour.merge(assignments[["location_group_id", "k3_cluster_id"]], on="location_group_id", how="inner")
    profile = location_hour.groupby(["k3_cluster_id", "hour"], as_index=False).agg(
        mean_hourly_share=("mean_day_share", "mean"),
        n_location_groups=("location_group_id", "nunique"),
    )
    all_hours = pd.MultiIndex.from_product(
        [sorted(assignments["k3_cluster_id"].unique()), range(24)],
        names=["k3_cluster_id", "hour"],
    )
    return profile.set_index(["k3_cluster_id", "hour"]).reindex(all_hours).reset_index()


def plot_hourly_profile(profile: pd.DataFrame, output: Path, title: str) -> None:
    """Plot hourly profile by k=3 cluster."""
    plt, PercentFormatter = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for cluster_id, rows in profile.groupby("k3_cluster_id", sort=True):
        cluster_id = int(cluster_id)
        ax.plot(
            rows["hour"],
            rows["mean_hourly_share"],
            marker="o",
            markersize=4.6,
            linewidth=2.3,
            color=COLORS.get(cluster_id),
            label=K3_CLUSTER_LABELS.get(cluster_id, f"cluster_{cluster_id}"),
        )
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average hourly share of daily count")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    clean_axes(ax, grid_axis="both")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    fig.subplots_adjust(bottom=0.27)
    save_figure(fig, output)
    plt.close(fig)


def monthly_profile(valid_days: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized monthly profile by k=3 cluster."""
    location_month = valid_days.groupby(["location_group_id", "month"], as_index=False).agg(
        mean_daily_total_month=("daily_total", "mean"),
        n_valid_days=("date", "nunique"),
    )
    location_overall = valid_days.groupby("location_group_id", as_index=False).agg(
        overall_mean_daily_total=("daily_total", "mean"),
    )
    location_month = location_month.merge(location_overall, on="location_group_id", how="left")
    location_month["normalized_monthly_profile"] = location_month["mean_daily_total_month"] / location_month["overall_mean_daily_total"]
    location_month = location_month.merge(assignments[["location_group_id", "k3_cluster_id"]], on="location_group_id", how="inner")
    profile = location_month.groupby(["k3_cluster_id", "month"], as_index=False).agg(
        mean_normalized_monthly_profile=("normalized_monthly_profile", "mean"),
        n_location_groups=("location_group_id", "nunique"),
    )
    all_months = pd.MultiIndex.from_product(
        [sorted(assignments["k3_cluster_id"].unique()), range(1, 13)],
        names=["k3_cluster_id", "month"],
    )
    return profile.set_index(["k3_cluster_id", "month"]).reindex(all_months).reset_index()


def plot_monthly_profile(profile: pd.DataFrame) -> None:
    """Plot normalized monthly profile by k=3 cluster."""
    plt, PercentFormatter = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for cluster_id, rows in profile.groupby("k3_cluster_id", sort=True):
        cluster_id = int(cluster_id)
        ax.plot(
            rows["month"],
            rows["mean_normalized_monthly_profile"],
            marker="o",
            markersize=4.8,
            linewidth=2.3,
            color=COLORS.get(cluster_id),
            label=K3_CLUSTER_LABELS.get(cluster_id, f"cluster_{cluster_id}"),
        )
    ax.axhline(1, color="#333333", linewidth=1, linestyle="--", alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Relative mean daily count")
    ax.set_title("Exploratory k=3 normalized monthly profile")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    clean_axes(ax, grid_axis="both")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    fig.subplots_adjust(bottom=0.27)
    save_figure(fig, FIG_MONTHLY)
    plt.close(fig)


def plot_transition_heatmap(transition: pd.DataFrame) -> None:
    """Plot k2 to k3 transition heatmap."""
    plt, _ = setup_matplotlib()
    table = transition.pivot(index="k2_cluster_id", columns="k3_cluster_id", values="n_location_groups").fillna(0)
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"k=3\ncluster {int(c)}" for c in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"k=2 cluster {int(i)}" for i in table.index])
    ax.set_xlabel("Exploratory k=3 cluster")
    ax.set_ylabel("Final k=2 cluster")
    ax.set_title("Transition from final k=2 to exploratory k=3")
    for i, k2_id in enumerate(table.index):
        for j, k3_id in enumerate(table.columns):
            value = int(table.loc[k2_id, k3_id])
            pct = transition.loc[
                (transition["k2_cluster_id"] == k2_id) & (transition["k3_cluster_id"] == k3_id),
                "pct_of_k2_cluster",
            ].iloc[0]
            label = f"{value}\n({pct:.0%})" if value else "0"
            ax.text(j, i, label, ha="center", va="center", color="#222222", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.85, label="Location groups")
    save_figure(fig, FIG_HEATMAP)
    plt.close(fig)


def run_exploration() -> dict[str, object]:
    """Run the k=3 exploratory workflow."""
    features, k2, groups = load_inputs()
    compact = create_compact5(features)
    k3_assignments, _scaled_frame, model = fit_k3(compact, groups)
    overall_silhouette = float(silhouette_score(k3_assignments[[f"z_{f}" for f in COMPACT5_FEATURES]], k3_assignments["k3_cluster_id"]))
    k3_summary = build_k3_summary(k3_assignments, overall_silhouette, float(model.inertia_))
    transition, info, merged = compare_k2_k3(k2, k3_assignments)
    broad_summary = build_broad_split_summary(merged, info["k2_broad_cluster_id"], info["k3_clusters_splitting_k2_broad"])
    diff = build_broad_feature_differences(broad_summary, merged, info["k2_broad_cluster_id"])

    for path in [K3_CLUSTERS_OUTPUT, TRANSITION_OUTPUT, K3_SUMMARY_OUTPUT, BROAD_SPLIT_OUTPUT, BROAD_DIFF_OUTPUT]:
        path.parent.mkdir(parents=True, exist_ok=True)
    k3_assignments.to_csv(K3_CLUSTERS_OUTPUT, index=False)
    transition.to_csv(TRANSITION_OUTPUT, index=False)
    k3_summary.to_csv(K3_SUMMARY_OUTPUT, index=False)
    broad_summary.to_csv(BROAD_SPLIT_OUTPUT, index=False)
    diff.to_csv(BROAD_DIFF_OUTPUT, index=False)

    plot_standardized_profile(k3_summary, FIG_K3_PROFILE, "Exploratory k=3 standardized compact5 profile")
    plot_standardized_profile(broad_summary.rename(columns={"k3_cluster_id": "k3_cluster_id"}), FIG_BROAD_PROFILE, "k=3 split within final k=2 broad cluster")
    counts = pd.read_parquet(COUNTS_INPUT)
    valid_days = build_valid_days(counts)
    plot_hourly_profile(location_level_hourly_profiles(counts, k3_assignments, valid_days, weekend=False), FIG_WEEKDAY, "Exploratory k=3 weekday hourly profile")
    plot_hourly_profile(location_level_hourly_profiles(counts, k3_assignments, valid_days, weekend=True), FIG_WEEKEND, "Exploratory k=3 weekend hourly profile")
    plot_monthly_profile(monthly_profile(valid_days, k3_assignments))
    plot_transition_heatmap(transition)

    return {
        "n_rows_model": int(len(k3_assignments)),
        "k3_cluster_sizes": k3_assignments["k3_cluster_id"].value_counts().sort_index().astype(int).to_dict(),
        "small_cluster_jaccard": info["small_cluster_jaccard"],
        "broad_split_clusters": info["k3_clusters_splitting_k2_broad"],
        "outputs": {
            "k3_clusters": str(K3_CLUSTERS_OUTPUT.relative_to(PROJECT_ROOT)),
            "transition": str(TRANSITION_OUTPUT.relative_to(PROJECT_ROOT)),
            "k3_summary": str(K3_SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)),
            "broad_split": str(BROAD_SPLIT_OUTPUT.relative_to(PROJECT_ROOT)),
            "feature_differences": str(BROAD_DIFF_OUTPUT.relative_to(PROJECT_ROOT)),
            "fig_k3_profile": str(FIG_K3_PROFILE.relative_to(PROJECT_ROOT)),
            "fig_broad_profile": str(FIG_BROAD_PROFILE.relative_to(PROJECT_ROOT)),
            "fig_weekday": str(FIG_WEEKDAY.relative_to(PROJECT_ROOT)),
            "fig_weekend": str(FIG_WEEKEND.relative_to(PROJECT_ROOT)),
            "fig_monthly": str(FIG_MONTHLY.relative_to(PROJECT_ROOT)),
            "fig_heatmap": str(FIG_HEATMAP.relative_to(PROJECT_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore global k=3 KMeans compared with final k=2.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        result = run_exploration()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Exploratory k=3 compact5 analysis complete")
    print(f"- rows model: {result['n_rows_model']}")
    print(f"- k3 cluster sizes: {result['k3_cluster_sizes']}")
    print(f"- small cluster Jaccard k2 vs k3: {result['small_cluster_jaccard']:.3f}")
    print(f"- broad split clusters: {result['broad_split_clusters']}")
    print(f"- outputs: {result['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
