"""Run KMeans elbow diagnostics on selected location-group features.

This script standardizes the selected clustering features and runs KMeans for
k = 1..10. It does not choose a final k automatically; the output is diagnostic
only.
"""

from __future__ import annotations

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
DEFAULT_FEATURE_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_selected.csv"
DEFAULT_SELECTED_FEATURES_INPUT = PROJECT_ROOT / "outputs" / "selected_features.json"
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "outputs" / "kmeans_elbow_results.csv"
DEFAULT_DIAGNOSTICS_OUTPUT = PROJECT_ROOT / "outputs" / "kmeans_elbow_diagnostics.json"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "outputs" / "kmeans_elbow_plot.png"
DEFAULT_SCALER_OUTPUT = PROJECT_ROOT / "outputs" / "scaler.joblib"

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

KMEANS_PARAMETERS = {
    "init": "k-means++",
    "n_init": 100,
    "max_iter": 500,
    "tol": 1e-4,
    "algorithm": "lloyd",
    "random_state": 42,
}


def load_selected_features(path: Path) -> list[str]:
    """Load selected clustering feature names from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Selected feature JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_features")
    if not isinstance(selected, list) or not selected:
        raise ValueError("selected_features.json must contain a non-empty 'selected_features' list.")
    if not all(isinstance(feature, str) for feature in selected):
        raise ValueError("All selected feature names must be strings.")
    return selected


def load_feature_table(path: Path) -> pd.DataFrame:
    """Load selected feature table."""
    if not path.exists():
        raise FileNotFoundError(f"Selected feature table not found: {path}")
    features = pd.read_csv(path)
    if "location_group_id" not in features.columns:
        raise ValueError("Selected feature table must include location_group_id for traceability.")
    return features


def validate_feature_matrix(features: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Validate selected features and return a clean numeric matrix."""
    forbidden = sorted(set(selected_features) & FORBIDDEN_CLUSTERING_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden variables included in clustering features: {', '.join(forbidden)}")

    missing = sorted(set(selected_features) - set(features.columns))
    if missing:
        raise ValueError(f"Selected feature table missing selected features: {', '.join(missing)}")

    matrix = features[selected_features].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    if matrix.isna().any().any():
        missing_counts = matrix.isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Selected feature matrix contains missing or infinite values: {missing_counts}")
    return matrix


def scale_features(matrix: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Apply StandardScaler to the selected feature matrix."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler


def run_elbow_models(scaled: np.ndarray, max_k: int = 10) -> pd.DataFrame:
    """Run KMeans for k=1..max_k and collect elbow diagnostics."""
    n_samples = scaled.shape[0]
    if n_samples < 2:
        raise ValueError("At least two rows are required for KMeans elbow diagnostics.")
    max_k = min(max_k, n_samples)
    records: list[dict[str, object]] = []
    previous_inertia: float | None = None

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, **KMEANS_PARAMETERS)
        labels = model.fit_predict(scaled)
        inertia = float(model.inertia_)
        if previous_inertia is None or previous_inertia == 0:
            relative_improvement = np.nan
        else:
            relative_improvement = (previous_inertia - inertia) / previous_inertia

        cluster_sizes = pd.Series(labels).value_counts().sort_index().astype(int).to_dict()
        if k >= 2 and k < n_samples:
            silhouette = float(silhouette_score(scaled, labels))
        else:
            silhouette = np.nan

        records.append(
            {
                "k": k,
                "inertia": inertia,
                "relative_inertia_improvement": relative_improvement,
                "n_iter": int(model.n_iter_),
                "reached_max_iter": bool(model.n_iter_ >= KMEANS_PARAMETERS["max_iter"]),
                "cluster_sizes": json.dumps({str(key): int(value) for key, value in cluster_sizes.items()}),
                "silhouette_score": silhouette,
            }
        )
        previous_inertia = inertia
    return pd.DataFrame.from_records(records)


def write_elbow_plot(results: pd.DataFrame, path: Path) -> None:
    """Write a diagnostic elbow plot to PNG."""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(results["k"], results["inertia"], marker="o", linewidth=2, color="#1f77b4")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("KMeans Elbow Diagnostic")
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
        "Diagnostic only: KMeans uses squared Euclidean distance in standardized feature space.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_diagnostics(
    features: pd.DataFrame,
    selected_features: list[str],
    matrix: pd.DataFrame,
    scaler: StandardScaler,
    results: pd.DataFrame,
) -> dict[str, object]:
    """Build machine-readable KMeans elbow diagnostics."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "feature_table": str(DEFAULT_FEATURE_INPUT.relative_to(PROJECT_ROOT)),
            "selected_features": str(DEFAULT_SELECTED_FEATURES_INPUT.relative_to(PROJECT_ROOT)),
        },
        "outputs": {
            "results": str(DEFAULT_RESULTS_OUTPUT.relative_to(PROJECT_ROOT)),
            "diagnostics": str(DEFAULT_DIAGNOSTICS_OUTPUT.relative_to(PROJECT_ROOT)),
            "plot": str(DEFAULT_PLOT_OUTPUT.relative_to(PROJECT_ROOT)),
            "scaler": str(DEFAULT_SCALER_OUTPUT.relative_to(PROJECT_ROOT)),
        },
        "n_rows": int(len(features)),
        "n_features": int(len(selected_features)),
        "selected_features": selected_features,
        "forbidden_variables_checked": sorted(FORBIDDEN_CLUSTERING_FEATURES),
        "k_range": [int(results["k"].min()), int(results["k"].max())],
        "kmeans_parameters": KMEANS_PARAMETERS,
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
        "methodological_notes": [
            "KMeans uses squared Euclidean distance in standardized feature space.",
            "No custom distance metric is used.",
            "Silhouette score is included as a secondary diagnostic only.",
            "No final k is selected automatically; inspect the elbow diagnostics before choosing k.",
        ],
        "final_k_recommendation": None,
    }


def run_kmeans_elbow(
    feature_input: Path = DEFAULT_FEATURE_INPUT,
    selected_features_input: Path = DEFAULT_SELECTED_FEATURES_INPUT,
    results_output: Path = DEFAULT_RESULTS_OUTPUT,
    diagnostics_output: Path = DEFAULT_DIAGNOSTICS_OUTPUT,
    plot_output: Path = DEFAULT_PLOT_OUTPUT,
    scaler_output: Path = DEFAULT_SCALER_OUTPUT,
    max_k: int = 10,
) -> dict[str, object]:
    """Run the full KMeans elbow diagnostic workflow."""
    selected_features = load_selected_features(selected_features_input)
    features = load_feature_table(feature_input)
    matrix = validate_feature_matrix(features, selected_features)
    scaled, scaler = scale_features(matrix)
    results = run_elbow_models(scaled, max_k=max_k)

    for path in [results_output, diagnostics_output, plot_output, scaler_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_output, index=False)
    joblib.dump(scaler, scaler_output)
    write_elbow_plot(results, plot_output)

    diagnostics = build_diagnostics(features, selected_features, matrix, scaler, results)
    diagnostics_output.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KMeans elbow diagnostics for location-group features.")
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_FEATURE_INPUT)
    parser.add_argument("--selected-features-input", type=Path, default=DEFAULT_SELECTED_FEATURES_INPUT)
    parser.add_argument("--max-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        diagnostics = run_kmeans_elbow(
            feature_input=args.feature_input,
            selected_features_input=args.selected_features_input,
            max_k=args.max_k,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("KMeans elbow diagnostics complete")
    print(f"- rows: {diagnostics['n_rows']}")
    print(f"- features: {diagnostics['selected_features']}")
    print(f"- k range: {diagnostics['k_range']}")
    print(f"- outputs: {diagnostics['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
