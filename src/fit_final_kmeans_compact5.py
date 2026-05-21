"""Final K-means model with the compact5 feature set.

This is the main model used in the project.  The script is deliberately kept
straightforward: load the candidate feature table, build compact5, standardise
the five features, fit K-means with k=2, and write the final tables.
"""

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

COMPACT5_SOURCE_COLUMNS = [
    "location_group_id",
    "log_weekend_weekday_ratio",
    "weekday_morning_peak_share",
    "weekday_evening_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "log_summer_winter_ratio",
]

# These columns may be useful for interpretation, but they must not drive K-means.
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

KMEANS_SETTINGS = {
    "n_clusters": 2,
    "init": "k-means++",
    "n_init": 100,
    "max_iter": 500,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "random_state": 42,
}


def read_input_tables(feature_input=FEATURE_INPUT, location_groups_input=LOCATION_GROUPS_INPUT):
    if not feature_input.exists():
        raise FileNotFoundError(f"Missing input file: {feature_input}")
    if not location_groups_input.exists():
        raise FileNotFoundError(f"Missing input file: {location_groups_input}")

    features = pd.read_csv(feature_input)
    location_groups = pd.read_csv(location_groups_input)

    missing = sorted(set(COMPACT5_SOURCE_COLUMNS) - set(features.columns))
    if missing:
        raise ValueError(f"Missing compact5 source columns: {', '.join(missing)}")

    return features, location_groups


def make_compact5(features):
    """Create the five final clustering features."""
    data = features.copy()

    numeric_columns = [column for column in COMPACT5_SOURCE_COLUMNS if column != "location_group_id"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["weekday_commute_peak_share"] = (
        data["weekday_morning_peak_share"] + data["weekday_evening_peak_share"]
    )
    data[COMPACT5_FEATURES] = data[COMPACT5_FEATURES].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=COMPACT5_FEATURES).copy()

    return data


def get_feature_matrix(compact):
    forbidden = sorted(set(COMPACT5_FEATURES) & FORBIDDEN_CLUSTERING_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden clustering features used: {', '.join(forbidden)}")

    matrix = compact[COMPACT5_FEATURES].apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any():
        missing_counts = matrix.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Missing values in final feature matrix: {missing_counts}")

    return matrix


def describe_centroid(z_means):
    """Short, cautious description based on the standardised cluster profile."""
    is_seasonal = (
        z_means.get("log_summer_winter_ratio", 0) > 1.0
        and z_means.get("log_weekend_weekday_ratio", 0) > 1.0
        and z_means.get("weekend_midday_afternoon_share", 0) > 1.0
    )
    if is_seasonal:
        return "strongly seasonal / recreational-like temporal pattern"
    return "broad mixed / less seasonal temporal pattern"


def top_features_text(z_means, n_features=5):
    ranked = z_means.reindex(z_means.abs().sort_values(ascending=False).index).head(n_features)
    text_parts = []
    for feature, value in ranked.items():
        direction = "high" if value >= 0 else "low"
        text_parts.append(f"{direction} {feature} ({value:.2f} SD)")
    return "; ".join(text_parts)


def build_assignment_table(compact, location_groups, matrix, scaled, labels, model):
    output = compact[["location_group_id"]].copy()
    output["cluster_id"] = labels.astype(int)
    output["cluster_name"] = output["cluster_id"].map(lambda cluster_id: f"cluster_{cluster_id}")
    output["distance_to_assigned_centroid"] = np.linalg.norm(
        scaled - model.cluster_centers_[labels], axis=1
    )
    output["silhouette_value"] = silhouette_samples(scaled, labels)

    for feature_number, feature in enumerate(COMPACT5_FEATURES):
        output[feature] = matrix[feature].to_numpy()
        output[f"z_{feature}"] = scaled[:, feature_number]

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
    available_columns = [column for column in metadata_columns if column in location_groups.columns]
    return output.merge(location_groups[available_columns], on="location_group_id", how="left")


def build_cluster_summary(assignments):
    records = []
    n_total = len(assignments)

    for cluster_id, rows in assignments.groupby("cluster_id", sort=True):
        z_means = pd.Series(
            {feature: float(rows[f"z_{feature}"].mean()) for feature in COMPACT5_FEATURES}
        )
        record = {
            "cluster_id": int(cluster_id),
            "cluster_name": f"cluster_{int(cluster_id)}",
            "n_observations": int(len(rows)),
            "pct_observations": float(len(rows) / n_total),
            "silhouette_mean": float(rows["silhouette_value"].mean()),
            "silhouette_median": float(rows["silhouette_value"].median()),
            "pct_negative_silhouette": float((rows["silhouette_value"] < 0).mean()),
            "top_distinctive_features_signed": top_features_text(z_means),
            "cautious_characterization": describe_centroid(z_means),
        }

        for feature in COMPACT5_FEATURES:
            record[f"raw_mean_{feature}"] = float(rows[feature].mean())
            record[f"raw_median_{feature}"] = float(rows[feature].median())
            record[f"z_mean_{feature}"] = float(rows[f"z_{feature}"].mean())

        records.append(record)

    return pd.DataFrame(records)


def build_silhouette_summary(assignments):
    records = []
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

    return pd.DataFrame(records)


def run_final_model(feature_input=FEATURE_INPUT, location_groups_input=LOCATION_GROUPS_INPUT):
    features, location_groups = read_input_tables(feature_input, location_groups_input)
    compact = make_compact5(features)
    matrix = get_feature_matrix(compact)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    model = KMeans(**KMEANS_SETTINGS)
    labels = model.fit_predict(scaled)

    assignments = build_assignment_table(compact, location_groups, matrix, scaled, labels, model)
    summary = build_cluster_summary(assignments)
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
        "rows_removed_missing": int(len(features) - len(compact)),
        "features": COMPACT5_FEATURES,
        "inertia": float(model.inertia_),
        "overall_silhouette": float(assignments["silhouette_value"].mean()),
        "cluster_sizes": assignments["cluster_id"].value_counts().sort_index().astype(int).to_dict(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the final K-means k=2 compact5 model.")
    parser.add_argument("--feature-input", type=Path, default=FEATURE_INPUT)
    parser.add_argument("--location-groups-input", type=Path, default=LOCATION_GROUPS_INPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = run_final_model(args.feature_input, args.location_groups_input)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Final K-means compact5 model complete")
    print(f"- rows input: {result['n_rows_input']}")
    print(f"- rows model: {result['n_rows_model']}")
    print(f"- rows removed because of missing compact5 values: {result['rows_removed_missing']}")
    print(f"- features: {result['features']}")
    print(f"- inertia: {result['inertia']:.3f}")
    print(f"- overall silhouette: {result['overall_silhouette']:.3f}")
    print(f"- cluster sizes: {result['cluster_sizes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
