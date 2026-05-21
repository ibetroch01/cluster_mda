"""Run KMeans feature-set sensitivity analysis for location-group features."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from plot_style import CLUSTER_COLORS, clean_axes, save_figure, setup_matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_candidate.csv"
DIAGNOSTICS_OUTPUT = PROJECT_ROOT / "outputs" / "feature_set_kmeans_diagnostics.csv"
MODEL_COMPARISON_OUTPUT = PROJECT_ROOT / "outputs" / "feature_set_model_comparison.csv"
CLUSTER_PROFILES_OUTPUT = PROJECT_ROOT / "outputs" / "feature_set_cluster_profiles.csv"
SENSITIVITY_FIG = PROJECT_ROOT / "outputs" / "fig_feature_set_sensitivity.png"

KMEANS_PARAMETERS = {
    "init": "k-means++",
    "n_init": 100,
    "max_iter": 500,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "random_state": 42,
}

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
    "long",
    "lat",
    "gemeente",
    "province",
    "site_id",
    "location_group_id",
}

FEATURE_SETS = {
    "full9": [
        "log_weekend_weekday_ratio",
        "weekday_morning_peak_share",
        "weekday_evening_peak_share",
        "weekday_midday_share",
        "weekend_midday_afternoon_share",
        "weekday_peak_concentration_2h",
        "log_summer_winter_ratio",
        "night_share",
        "daily_variability_cv",
    ],
    "core7_no_optional": [
        "log_weekend_weekday_ratio",
        "weekday_morning_peak_share",
        "weekday_evening_peak_share",
        "weekday_midday_share",
        "weekend_midday_afternoon_share",
        "weekday_peak_concentration_2h",
        "log_summer_winter_ratio",
    ],
    "compact5": [
        "log_weekend_weekday_ratio",
        "weekday_commute_peak_share",
        "weekday_midday_share",
        "weekend_midday_afternoon_share",
        "log_summer_winter_ratio",
    ],
}


def load_features(path: Path) -> pd.DataFrame:
    """Load candidate features and create compact derived feature."""
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    features = pd.read_csv(path)
    required = {
        "location_group_id",
        "weekday_morning_peak_share",
        "weekday_evening_peak_share",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature table missing required columns: {', '.join(missing)}")
    features = features.copy()
    features["weekday_commute_peak_share"] = (
        pd.to_numeric(features["weekday_morning_peak_share"], errors="coerce")
        + pd.to_numeric(features["weekday_evening_peak_share"], errors="coerce")
    )
    return features


def validate_feature_set(features: pd.DataFrame, feature_set_name: str, columns: list[str]) -> pd.DataFrame:
    """Validate a feature set and return a complete numeric matrix."""
    forbidden = sorted(set(columns) & FORBIDDEN_CLUSTERING_FEATURES)
    if forbidden:
        raise AssertionError(
            f"Feature set {feature_set_name} includes forbidden clustering variables: {', '.join(forbidden)}"
        )
    missing = sorted(set(columns) - set(features.columns))
    if missing:
        raise ValueError(f"Feature set {feature_set_name} missing columns: {', '.join(missing)}")
    matrix = features[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return matrix.dropna().copy()


def aligned_rows(features: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Return feature table rows aligned to the complete matrix."""
    return features.loc[matrix.index].copy()


def run_elbow(feature_set_name: str, matrix: pd.DataFrame) -> pd.DataFrame:
    """Run KMeans k=1..10 for one feature set."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    records: list[dict[str, object]] = []
    previous_inertia: float | None = None
    max_k = min(10, len(matrix))
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, **KMEANS_PARAMETERS)
        labels = model.fit_predict(scaled)
        inertia = float(model.inertia_)
        relative_improvement = np.nan if previous_inertia in (None, 0) else (previous_inertia - inertia) / previous_inertia
        sizes = pd.Series(labels).value_counts().sort_index().astype(int)
        silhouette = float(silhouette_score(scaled, labels)) if k >= 2 and k < len(matrix) else np.nan
        records.append(
            {
                "feature_set": feature_set_name,
                "n_rows": int(len(matrix)),
                "n_features": int(matrix.shape[1]),
                "k": k,
                "inertia": inertia,
                "relative_inertia_improvement": relative_improvement,
                "silhouette_score_overall": silhouette,
                "cluster_sizes": json.dumps({str(key): int(value) for key, value in sizes.items()}),
                "smallest_cluster_pct": float(sizes.min() / len(matrix)),
                "n_iter": int(model.n_iter_),
                "reached_max_iter": bool(model.n_iter_ >= KMEANS_PARAMETERS["max_iter"]),
            }
        )
        previous_inertia = inertia
    return pd.DataFrame.from_records(records)


def fit_candidate_models(
    feature_set_name: str,
    feature_rows: pd.DataFrame,
    matrix: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit k=2 and k=3 and return model comparison plus cluster profiles."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    z_columns = [f"z_{feature}" for feature in selected_features]
    model_records: list[dict[str, object]] = []
    profile_records: list[dict[str, object]] = []

    for k in [2, 3]:
        model = KMeans(n_clusters=k, **KMEANS_PARAMETERS)
        labels = model.fit_predict(scaled)
        silhouette_values = silhouette_samples(scaled, labels)
        overall_silhouette = float(silhouette_score(scaled, labels))
        sizes = pd.Series(labels).value_counts().sort_index().astype(int)
        model_records.append(
            {
                "feature_set": feature_set_name,
                "k": k,
                "n_rows": int(len(matrix)),
                "n_features": int(matrix.shape[1]),
                "inertia": float(model.inertia_),
                "silhouette_score_overall": overall_silhouette,
                "cluster_sizes": json.dumps({str(key): int(value) for key, value in sizes.items()}),
                "smallest_cluster_pct": float(sizes.min() / len(matrix)),
                "pct_negative_silhouette": float((silhouette_values < 0).mean()),
                "n_iter": int(model.n_iter_),
                "reached_max_iter": bool(model.n_iter_ >= KMEANS_PARAMETERS["max_iter"]),
            }
        )

        assignments = pd.DataFrame(
            {
                "location_group_id": feature_rows["location_group_id"].astype(str).to_numpy(),
                "cluster_id": labels.astype(int),
                "silhouette_value": silhouette_values,
            },
            index=matrix.index,
        )
        z_frame = pd.DataFrame(scaled, columns=z_columns, index=matrix.index)
        assignments = pd.concat([assignments, matrix, z_frame], axis=1)
        for cluster_id, cluster_rows in assignments.groupby("cluster_id", sort=True):
            values = cluster_rows["silhouette_value"]
            record: dict[str, object] = {
                "feature_set": feature_set_name,
                "k": k,
                "cluster_id": int(cluster_id),
                "cluster_name": f"cluster_{int(cluster_id)}",
                "n_observations": int(len(cluster_rows)),
                "pct_observations": float(len(cluster_rows) / len(assignments)),
                "silhouette_mean": float(values.mean()),
                "silhouette_median": float(values.median()),
                "n_negative_silhouette": int((values < 0).sum()),
                "pct_negative_silhouette": float((values < 0).mean()),
            }
            z_means = cluster_rows[z_columns].mean()
            z_means.index = selected_features
            top = z_means.reindex(z_means.abs().sort_values(ascending=False).index).head(5)
            record["top_5_signed_centroid_features"] = "; ".join(
                f"{'high' if value >= 0 else 'low'} {feature} ({value:.2f} SD)"
                for feature, value in top.items()
            )
            for feature in selected_features:
                record[f"raw_mean_{feature}"] = float(cluster_rows[feature].mean())
                record[f"z_centroid_{feature}"] = float(cluster_rows[f"z_{feature}"].mean())
            profile_records.append(record)
    return pd.DataFrame.from_records(model_records), pd.DataFrame.from_records(profile_records)


def plot_sensitivity(model_comparison: pd.DataFrame) -> None:
    """Plot silhouette scores for the feature-set sensitivity check."""
    plt, _ = setup_matplotlib()
    plot_data = model_comparison.sort_values(["feature_set", "k"]).copy()
    feature_sets = ["full9", "core7_no_optional", "compact5"]
    k_values = sorted(plot_data["k"].unique())
    x = np.arange(len(feature_sets))
    width = 0.34
    colors = {2: CLUSTER_COLORS["regular"], 3: CLUSTER_COLORS["seasonal"]}

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for index, k in enumerate(k_values):
        subset = plot_data.loc[plot_data["k"] == k].set_index("feature_set").reindex(feature_sets)
        offset = (index - (len(k_values) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            subset["silhouette_score_overall"],
            width=width,
            color=colors.get(k, CLUSTER_COLORS["grey"]),
            label=f"k = {k}",
            alpha=0.95,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["full9", "core7\n(no optional)", "compact5"])
    ax.set_ylim(0, max(0.68, plot_data["silhouette_score_overall"].max() + 0.08))
    ax.set_ylabel("Overall silhouette score")
    ax.set_title("Feature-set sensitivity of candidate K-means models")
    clean_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, SENSITIVITY_FIG)
    plt.close(fig)


def run_sensitivity(input_path: Path = DEFAULT_INPUT) -> dict[str, object]:
    """Run the full feature-set sensitivity workflow."""
    features = load_features(input_path)
    all_diagnostics: list[pd.DataFrame] = []
    all_model_comparisons: list[pd.DataFrame] = []
    all_profiles: list[pd.DataFrame] = []

    for feature_set_name, columns in FEATURE_SETS.items():
        matrix = validate_feature_set(features, feature_set_name, columns)
        rows = aligned_rows(features, matrix)
        all_diagnostics.append(run_elbow(feature_set_name, matrix))
        comparison, profiles = fit_candidate_models(feature_set_name, rows, matrix, columns)
        all_model_comparisons.append(comparison)
        all_profiles.append(profiles)

    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    model_comparison = pd.concat(all_model_comparisons, ignore_index=True)
    cluster_profiles = pd.concat(all_profiles, ignore_index=True)

    for path in [DIAGNOSTICS_OUTPUT, MODEL_COMPARISON_OUTPUT, CLUSTER_PROFILES_OUTPUT, SENSITIVITY_FIG]:
        path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(DIAGNOSTICS_OUTPUT, index=False)
    model_comparison.to_csv(MODEL_COMPARISON_OUTPUT, index=False)
    cluster_profiles.to_csv(CLUSTER_PROFILES_OUTPUT, index=False)
    plot_sensitivity(model_comparison)

    return {
        "n_rows_input": int(len(features)),
        "feature_sets": FEATURE_SETS,
        "outputs": {
            "diagnostics": str(DIAGNOSTICS_OUTPUT.relative_to(PROJECT_ROOT)),
            "model_comparison": str(MODEL_COMPARISON_OUTPUT.relative_to(PROJECT_ROOT)),
            "cluster_profiles": str(CLUSTER_PROFILES_OUTPUT.relative_to(PROJECT_ROOT)),
            "sensitivity_figure": str(SENSITIVITY_FIG.relative_to(PROJECT_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KMeans feature-set sensitivity analysis.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_sensitivity(args.input)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Feature-set sensitivity analysis complete")
    print(f"- rows input: {result['n_rows_input']}")
    print(f"- feature sets: {list(result['feature_sets'])}")
    print(f"- outputs: {result['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
