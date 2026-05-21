"""Elbow diagnostics for the selected clustering features.

This script is used before the final model is chosen.  It standardises the
selected features and runs K-means for k = 1, ..., 10.  The output is diagnostic:
the script does not automatically choose k.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_selected.csv"
SELECTED_FEATURES_INPUT = PROJECT_ROOT / "outputs" / "selected_features.json"
RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "kmeans_elbow_results.csv"
DIAGNOSTICS_OUTPUT = PROJECT_ROOT / "outputs" / "kmeans_elbow_diagnostics.json"
PLOT_OUTPUT = PROJECT_ROOT / "outputs" / "kmeans_elbow_plot.png"
SCALER_OUTPUT = PROJECT_ROOT / "outputs" / "scaler.joblib"

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
    "gemeente",
    "site_id",
    "location_group_id",
}

KMEANS_SETTINGS = {
    "init": "k-means++",
    "n_init": 100,
    "max_iter": 500,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "random_state": 42,
}


def load_selected_features(path):
    if not path.exists():
        raise FileNotFoundError(f"Selected feature JSON not found: {path}")

    info = json.loads(path.read_text(encoding="utf-8"))
    features = info.get("selected_features")
    if not isinstance(features, list) or not features:
        raise ValueError("selected_features.json must contain a non-empty 'selected_features' list.")

    return features


def load_feature_matrix(feature_path, selected_features):
    if not feature_path.exists():
        raise FileNotFoundError(f"Selected feature table not found: {feature_path}")

    data = pd.read_csv(feature_path)
    if "location_group_id" not in data.columns:
        raise ValueError("The feature table must include location_group_id for traceability.")

    forbidden = sorted(set(selected_features) & FORBIDDEN_CLUSTERING_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden variables included in clustering features: {', '.join(forbidden)}")

    missing = sorted(set(selected_features) - set(data.columns))
    if missing:
        raise ValueError(f"Selected features missing from table: {', '.join(missing)}")

    matrix = data[selected_features].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    if matrix.isna().any().any():
        missing_counts = matrix.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Missing or infinite values in selected features: {missing_counts}")

    return data, matrix


def run_elbow(scaled_features, max_k=10):
    n_rows = scaled_features.shape[0]
    if n_rows < 2:
        raise ValueError("At least two rows are required for K-means.")

    max_k = min(max_k, n_rows)
    records = []
    previous_inertia = None

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, **KMEANS_SETTINGS)
        labels = model.fit_predict(scaled_features)
        inertia = float(model.inertia_)

        if previous_inertia is None or previous_inertia == 0:
            relative_improvement = np.nan
        else:
            relative_improvement = (previous_inertia - inertia) / previous_inertia

        if 2 <= k < n_rows:
            silhouette = float(silhouette_score(scaled_features, labels))
        else:
            silhouette = np.nan

        cluster_sizes = pd.Series(labels).value_counts().sort_index().astype(int).to_dict()
        records.append(
            {
                "k": k,
                "inertia": inertia,
                "relative_inertia_improvement": relative_improvement,
                "n_iter": int(model.n_iter_),
                "reached_max_iter": bool(model.n_iter_ >= KMEANS_SETTINGS["max_iter"]),
                "cluster_sizes": json.dumps({str(key): int(value) for key, value in cluster_sizes.items()}),
                "silhouette_score": silhouette,
            }
        )
        previous_inertia = inertia

    return pd.DataFrame(records)


def save_elbow_plot(results, output_path):
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(results["k"], results["inertia"], marker="o", linewidth=2, color="#1f77b4")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("K-means elbow diagnostic")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        results["k"],
        results["relative_inertia_improvement"],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        color="#d62728",
    )
    ax2.set_ylabel("Relative inertia improvement")

    fig.text(
        0.5,
        0.01,
        "Diagnostic only: K-means uses squared Euclidean distance after standardisation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_diagnostics(feature_table, selected_features, matrix, scaler, results):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "feature_table": str(FEATURE_INPUT.relative_to(PROJECT_ROOT)),
            "selected_features": str(SELECTED_FEATURES_INPUT.relative_to(PROJECT_ROOT)),
        },
        "outputs": {
            "results": str(RESULTS_OUTPUT.relative_to(PROJECT_ROOT)),
            "diagnostics": str(DIAGNOSTICS_OUTPUT.relative_to(PROJECT_ROOT)),
            "plot": str(PLOT_OUTPUT.relative_to(PROJECT_ROOT)),
            "scaler": str(SCALER_OUTPUT.relative_to(PROJECT_ROOT)),
        },
        "n_rows": int(len(feature_table)),
        "n_features": int(len(selected_features)),
        "selected_features": selected_features,
        "forbidden_variables_checked": sorted(FORBIDDEN_CLUSTERING_FEATURES),
        "k_range": [int(results["k"].min()), int(results["k"].max())],
        "kmeans_parameters": KMEANS_SETTINGS,
        "standard_scaler": {
            "feature_names": selected_features,
            "mean": {feature: float(value) for feature, value in zip(selected_features, scaler.mean_)},
            "scale": {feature: float(value) for feature, value in zip(selected_features, scaler.scale_)},
        },
        "feature_summary_before_scaling": {
            feature: {
                "mean": float(matrix[feature].mean()),
                "std": float(matrix[feature].std(ddof=0)),
                "min": float(matrix[feature].min()),
                "max": float(matrix[feature].max()),
            }
            for feature in selected_features
        },
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
    }


def run_diagnostics(feature_input=FEATURE_INPUT, selected_features_input=SELECTED_FEATURES_INPUT):
    selected_features = load_selected_features(selected_features_input)
    feature_table, matrix = load_feature_matrix(feature_input, selected_features)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    results = run_elbow(scaled)

    for path in [RESULTS_OUTPUT, DIAGNOSTICS_OUTPUT, PLOT_OUTPUT, SCALER_OUTPUT]:
        path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(RESULTS_OUTPUT, index=False)
    save_elbow_plot(results, PLOT_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)

    diagnostics = build_diagnostics(feature_table, selected_features, matrix, scaler, results)
    DIAGNOSTICS_OUTPUT.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Run K-means elbow diagnostics.")
    parser.add_argument("--feature-input", type=Path, default=FEATURE_INPUT)
    parser.add_argument("--selected-features-input", type=Path, default=SELECTED_FEATURES_INPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        diagnostics = run_diagnostics(args.feature_input, args.selected_features_input)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("K-means elbow diagnostics complete")
    print(f"- rows used: {diagnostics['n_rows']}")
    print(f"- selected features: {diagnostics['selected_features']}")
    print(f"- k range: {diagnostics['k_range']}")
    print(f"- results: {RESULTS_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"- plot: {PLOT_OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
