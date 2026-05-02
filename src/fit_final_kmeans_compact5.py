"""Fit the final KMeans model for AWV location groups using compact5 features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_candidate.csv"
LOCATION_GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
CLUSTERS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
SUMMARY_OUTPUT = PROJECT_ROOT / "outputs" / "final_cluster_summary_k2_compact5.csv"
SILHOUETTE_OUTPUT = PROJECT_ROOT / "outputs" / "final_cluster_silhouette_k2_compact5.csv"
MODEL_OUTPUT = PROJECT_ROOT / "outputs" / "final_kmeans_model_k2_compact5.joblib"
SCALER_OUTPUT = PROJECT_ROOT / "outputs" / "final_scaler_k2_compact5.joblib"

COMPACT5_FEATURES = [
    "log_weekend_weekday_ratio",
    "weekday_commute_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "log_summer_winter_ratio",
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
    "long",
    "lat",
    "gemeente",
    "province",
    "site_id",
    "location_group_id",
}

KMEANS_PARAMETERS = {
    "n_clusters": 2,
    "init": "k-means++",
    "n_init": 100,
    "max_iter": 500,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "random_state": 42,
}


def load_inputs(feature_input: Path, location_groups_input: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load candidate features and optional location-group metadata."""
    if not feature_input.exists():
        raise FileNotFoundError(f"Missing input: {feature_input}")
    if not location_groups_input.exists():
        raise FileNotFoundError(f"Missing input: {location_groups_input}")
    features = pd.read_csv(feature_input)
    location_groups = pd.read_csv(location_groups_input)
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
        raise ValueError(f"Feature input missing required compact5 source columns: {', '.join(missing)}")
    return features, location_groups


def create_compact5(features: pd.DataFrame) -> pd.DataFrame:
    """Create compact5 feature table and remove rows with missing values."""
    output = features.copy()
    for column in [
        "log_weekend_weekday_ratio",
        "weekday_morning_peak_share",
        "weekday_evening_peak_share",
        "weekday_midday_share",
        "weekend_midday_afternoon_share",
        "log_summer_winter_ratio",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["weekday_commute_peak_share"] = (
        output["weekday_morning_peak_share"] + output["weekday_evening_peak_share"]
    )
    output[COMPACT5_FEATURES] = output[COMPACT5_FEATURES].replace([np.inf, -np.inf], np.nan)
    return output.dropna(subset=COMPACT5_FEATURES).copy()


def validate_features(selected_features: list[str], feature_table: pd.DataFrame) -> pd.DataFrame:
    """Assert no forbidden variables are used and return the numeric matrix."""
    forbidden = sorted(set(selected_features) & FORBIDDEN_CLUSTERING_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden variables included as clustering features: {', '.join(forbidden)}")
    missing = sorted(set(selected_features) - set(feature_table.columns))
    if missing:
        raise ValueError(f"Missing compact5 feature columns: {', '.join(missing)}")
    matrix = feature_table[selected_features].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        missing_counts = matrix.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Compact5 matrix contains missing values after filtering: {missing_counts}")
    return matrix


def distance_to_centroid(scaled: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance to the assigned centroid in standardized space."""
    return np.linalg.norm(scaled - centroids[labels], axis=1)


def signed_top_features(z_means: pd.Series, n: int = 5) -> str:
    """Return signed distinctive features from standardized centroid values."""
    top = z_means.reindex(z_means.abs().sort_values(ascending=False).index).head(n)
    return "; ".join(
        f"{'high' if value >= 0 else 'low'} {feature} ({value:.2f} SD)"
        for feature, value in top.items()
    )


def characterize_cluster(z_means: pd.Series) -> str:
    """Create a cautious temporal characterization from compact5 centroid values."""
    if (
        z_means.get("log_summer_winter_ratio", 0) > 1.0
        and z_means.get("log_weekend_weekday_ratio", 0) > 1.0
        and z_means.get("weekend_midday_afternoon_share", 0) > 1.0
    ):
        return "strongly seasonal / recreational-like temporal pattern"
    return "broad mixed / less seasonal temporal pattern"


def build_assignments(
    feature_table: pd.DataFrame,
    location_groups: pd.DataFrame,
    matrix: pd.DataFrame,
    scaled: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
) -> pd.DataFrame:
    """Build final per-location-group assignment table."""
    silhouette_values = silhouette_samples(scaled, labels)
    assignments = feature_table[["location_group_id"]].copy()
    assignments["cluster_id"] = labels.astype(int)
    assignments["cluster_name"] = assignments["cluster_id"].map(lambda value: f"cluster_{value}")
    assignments["distance_to_assigned_centroid"] = distance_to_centroid(scaled, labels, model.cluster_centers_)
    assignments["silhouette_value"] = silhouette_values
    for feature in COMPACT5_FEATURES:
        assignments[feature] = matrix[feature].to_numpy()
        assignments[f"z_{feature}"] = scaled[:, COMPACT5_FEATURES.index(feature)]

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
    available = [column for column in metadata_columns if column in location_groups.columns]
    assignments = assignments.merge(location_groups[available], on="location_group_id", how="left")
    return assignments


def build_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """Build final cluster summary."""
    records: list[dict[str, object]] = []
    n_total = len(assignments)
    for cluster_id, rows in assignments.groupby("cluster_id", sort=True):
        z_means = pd.Series(
            {feature: float(rows[f"z_{feature}"].mean()) for feature in COMPACT5_FEATURES}
        )
        record: dict[str, object] = {
            "cluster_id": int(cluster_id),
            "cluster_name": f"cluster_{int(cluster_id)}",
            "n_observations": int(len(rows)),
            "pct_observations": float(len(rows) / n_total),
            "silhouette_mean": float(rows["silhouette_value"].mean()),
            "silhouette_median": float(rows["silhouette_value"].median()),
            "pct_negative_silhouette": float((rows["silhouette_value"] < 0).mean()),
            "top_distinctive_features_signed": signed_top_features(z_means),
            "cautious_characterization": characterize_cluster(z_means),
        }
        for feature in COMPACT5_FEATURES:
            record[f"raw_mean_{feature}"] = float(rows[feature].mean())
            record[f"raw_median_{feature}"] = float(rows[feature].median())
            record[f"z_mean_{feature}"] = float(rows[f"z_{feature}"].mean())
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_silhouette_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """Build final silhouette summary per cluster."""
    records: list[dict[str, object]] = []
    n_total = len(assignments)
    for cluster_id, rows in assignments.groupby("cluster_id", sort=True):
        values = rows["silhouette_value"]
        records.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_name": f"cluster_{int(cluster_id)}",
                "n_observations": int(len(rows)),
                "pct_observations": float(len(rows) / n_total),
                "silhouette_mean": float(values.mean()),
                "silhouette_median": float(values.median()),
                "silhouette_min": float(values.min()),
                "silhouette_max": float(values.max()),
                "n_negative_silhouette": int((values < 0).sum()),
                "pct_negative_silhouette": float((values < 0).mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def run_final_model(
    feature_input: Path = FEATURE_INPUT,
    location_groups_input: Path = LOCATION_GROUPS_INPUT,
) -> dict[str, object]:
    """Fit and save the final compact5 KMeans model."""
    features, location_groups = load_inputs(feature_input, location_groups_input)
    compact = create_compact5(features)
    rows_removed_missing = int(len(features) - len(compact))
    matrix = validate_features(COMPACT5_FEATURES, compact)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    model = KMeans(**KMEANS_PARAMETERS)
    labels = model.fit_predict(scaled)

    assignments = build_assignments(compact, location_groups, matrix, scaled, labels, model)
    summary = build_summary(assignments)
    silhouette_summary = build_silhouette_summary(assignments)

    for path in [CLUSTERS_OUTPUT, SUMMARY_OUTPUT, SILHOUETTE_OUTPUT, MODEL_OUTPUT, SCALER_OUTPUT]:
        path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(CLUSTERS_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    silhouette_summary.to_csv(SILHOUETTE_OUTPUT, index=False)
    joblib.dump(model, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)

    return {
        "n_rows_input": int(len(features)),
        "n_rows_model": int(len(assignments)),
        "rows_removed_missing": rows_removed_missing,
        "features": COMPACT5_FEATURES,
        "inertia": float(model.inertia_),
        "overall_silhouette": float(assignments["silhouette_value"].mean()),
        "cluster_sizes": assignments["cluster_id"].value_counts().sort_index().astype(int).to_dict(),
        "outputs": {
            "clusters": str(CLUSTERS_OUTPUT.relative_to(PROJECT_ROOT)),
            "summary": str(SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)),
            "silhouette": str(SILHOUETTE_OUTPUT.relative_to(PROJECT_ROOT)),
            "model": str(MODEL_OUTPUT.relative_to(PROJECT_ROOT)),
            "scaler": str(SCALER_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit final KMeans k=2 compact5 model.")
    parser.add_argument("--feature-input", type=Path, default=FEATURE_INPUT)
    parser.add_argument("--location-groups-input", type=Path, default=LOCATION_GROUPS_INPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_final_model(args.feature_input, args.location_groups_input)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Final KMeans compact5 model complete")
    print(f"- rows input: {result['n_rows_input']}")
    print(f"- rows model: {result['n_rows_model']}")
    print(f"- features: {result['features']}")
    print(f"- inertia: {result['inertia']:.3f}")
    print(f"- overall silhouette: {result['overall_silhouette']:.3f}")
    print(f"- cluster sizes: {result['cluster_sizes']}")
    print(f"- outputs: {result['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
