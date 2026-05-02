"""Audit PCA and centroid-distance outliers in the final compact5 k=2 model."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_INPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
QUALITY_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_quality_summary.csv"
GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
AUDIT_OUTPUT = PROJECT_ROOT / "outputs" / "final_pca_outlier_audit.csv"
REPORT_OUTPUT = PROJECT_ROOT / "reports" / "final_pca_outlier_audit.md"

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
    "morning_07_09_coverage_ratio",
    "evening_16_18_coverage_ratio",
    "weekday_midday_10_15_coverage_ratio",
    "weekend_midday_afternoon_11_17_coverage_ratio",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required inputs."""
    for path in [CLUSTERS_INPUT, QUALITY_INPUT, GROUPS_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(CLUSTERS_INPUT), pd.read_csv(QUALITY_INPUT), pd.read_csv(GROUPS_INPUT)


def compute_pca(clusters: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    """Recompute PCA on standardized compact5 feature matrix."""
    z_features = [f"z_{feature}" for feature in COMPACT5_FEATURES]
    missing = sorted(set(z_features) - set(clusters.columns))
    if missing:
        raise ValueError(f"Final cluster table missing standardized features: {', '.join(missing)}")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(clusters[z_features])
    output = clusters.copy()
    output["pc1"] = coords[:, 0]
    output["pc2"] = coords[:, 1]
    output["abs_pc2"] = output["pc2"].abs()
    output["pc2_rank_desc"] = output["abs_pc2"].rank(ascending=False, method="min").astype(int)
    output["centroid_distance_rank_desc"] = output["distance_to_assigned_centroid"].rank(
        ascending=False,
        method="min",
    ).astype(int)
    return output, [float(value) for value in pca.explained_variance_ratio_]


def quality_flag(row: pd.Series) -> tuple[str, str]:
    """Flag whether a row looks like a possible data-quality issue."""
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
        ("morning_07_09_coverage_ratio < 0.90", row["morning_07_09_coverage_ratio"] < 0.90),
        ("evening_16_18_coverage_ratio < 0.90", row["evening_16_18_coverage_ratio"] < 0.90),
        ("weekday_midday_10_15_coverage_ratio < 0.90", row["weekday_midday_10_15_coverage_ratio"] < 0.90),
        (
            "weekend_midday_afternoon_11_17_coverage_ratio < 0.90",
            row["weekend_midday_afternoon_11_17_coverage_ratio"] < 0.90,
        ),
    ]
    reasons = [reason for reason, failed in checks if bool(failed)]
    if reasons:
        return "possible_data_quality_issue", "; ".join(reasons)
    return "valid_unusual_temporal_profile", "quality metrics sufficient under audit thresholds"


def build_audit(clusters: pd.DataFrame, quality: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Build outlier audit table for top |PC2| and centroid-distance observations."""
    top_pc2_ids = set(clusters.nlargest(5, "abs_pc2")["location_group_id"])
    top_distance_ids = set(clusters.nlargest(5, "distance_to_assigned_centroid")["location_group_id"])
    audit_ids = top_pc2_ids | top_distance_ids

    audit = clusters.loc[clusters["location_group_id"].isin(audit_ids)].copy()
    audit["outlier_reason"] = audit["location_group_id"].map(
        lambda location_group_id: "; ".join(
            reason
            for reason, included in [
                ("top_5_abs_pc2", location_group_id in top_pc2_ids),
                ("top_5_centroid_distance", location_group_id in top_distance_ids),
            ]
            if included
        )
    )

    group_columns = [
        "location_group_id",
        "gemeente_values",
        "site_ids",
        "n_sites_in_group",
        "mean_longitude",
        "mean_latitude",
    ]
    group_columns = [column for column in group_columns if column in groups.columns]
    audit = audit.drop(columns=[column for column in group_columns if column != "location_group_id" and column in audit.columns])
    audit = audit.merge(groups[group_columns], on="location_group_id", how="left")
    audit = audit.merge(quality[["location_group_id", *QUALITY_COLUMNS]], on="location_group_id", how="left")

    flags = audit.apply(quality_flag, axis=1, result_type="expand")
    audit["outlier_assessment"] = flags[0]
    audit["assessment_reason"] = flags[1]
    audit = audit.rename(columns={"gemeente_values": "gemeente"})

    ordered = [
        "location_group_id",
        "cluster_id",
        "cluster_name",
        "gemeente",
        "site_ids",
        *COMPACT5_FEATURES,
        "pc1",
        "pc2",
        "abs_pc2",
        "pc2_rank_desc",
        "distance_to_assigned_centroid",
        "centroid_distance_rank_desc",
        "silhouette_value",
        *QUALITY_COLUMNS,
        "outlier_reason",
        "outlier_assessment",
        "assessment_reason",
    ]
    return audit[ordered].sort_values(
        ["outlier_assessment", "pc2_rank_desc", "centroid_distance_rank_desc", "location_group_id"]
    )


def write_report(audit: pd.DataFrame, explained_variance: list[float]) -> None:
    """Write Dutch outlier audit report."""
    n_quality = int((audit["outlier_assessment"] == "possible_data_quality_issue").sum())
    n_valid = int((audit["outlier_assessment"] == "valid_unusual_temporal_profile").sum())
    lines = [
        "# PCA-outlieraudit finale k=2 compact5 clustering",
        "",
        f"Gegenereerd op: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Methode",
        "",
        "PCA werd opnieuw berekend op de gestandaardiseerde compact5-featurematrix van het finale k=2-model. "
        "Daarna zijn twee outlierlijsten gemaakt: de vijf grootste absolute PC2-scores en de vijf grootste afstanden tot de toegewezen K-means-centroid.",
        "",
        f"- PC1 verklaarde `{explained_variance[0] * 100:.1f}%` van de variantie.",
        f"- PC2 verklaarde `{explained_variance[1] * 100:.1f}%` van de variantie.",
        f"- Samen tonen PC1 en PC2 `{sum(explained_variance) * 100:.1f}%` van de gestandaardiseerde compact5-variantie.",
        "",
        "## Auditbeoordeling",
        "",
        f"- Aantal unieke outliers in de audit: `{len(audit)}`",
        f"- Lijken geldig ongebruikelijk temporeel profiel: `{n_valid}`",
        f"- Mogelijke data-quality issue: `{n_quality}`",
        "",
    ]
    if n_quality == 0:
        lines.append(
            "Op basis van de beschikbare kwaliteitsmetrics lijken deze PCA/centroid-outliers eerder geldige ongebruikelijke temporele profielen dan duidelijke data-quality problemen."
        )
    else:
        lines.append(
            "Een deel van de outliers heeft zwakkere kwaliteitsmetrics. Die locaties moeten manueel gecontroleerd worden voordat ze inhoudelijk zwaar worden geïnterpreteerd."
        )

    lines.extend(
        [
            "",
            "## Outliers",
            "",
            "| location_group_id | cluster | gemeente | reason | PC1 | PC2 | distance | silhouette | assessment |",
            "|---|---:|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in audit.itertuples(index=False):
        lines.append(
            "| "
            f"{row.location_group_id} | "
            f"{int(row.cluster_id)} | "
            f"{row.gemeente} | "
            f"{row.outlier_reason} | "
            f"{row.pc1:.3f} | "
            f"{row.pc2:.3f} | "
            f"{row.distance_to_assigned_centroid:.3f} | "
            f"{row.silhouette_value:.3f} | "
            f"{row.outlier_assessment} |"
        )

    lines.extend(
        [
            "",
            "## Voorzichtige conclusie",
            "",
            "Deze audit toont welke locatiegroepen extreem liggen in de PCA-weergave of relatief ver van hun centroid liggen. "
            "Dat betekent niet automatisch dat ze fout zijn. Wanneer kwaliteitsindicatoren voldoende zijn, worden ze hier gezien als geldige maar ongebruikelijke temporele profielen. "
            "De clusters blijven patronen in fietstellingen, geen bewezen fietsmotieven.",
            "",
        ]
    )
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def run_audit() -> dict[str, object]:
    """Run final PCA outlier audit."""
    clusters, quality, groups = load_inputs()
    clusters_with_pca, explained = compute_pca(clusters)
    audit = build_audit(clusters_with_pca, quality, groups)
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_OUTPUT, index=False)
    write_report(audit, explained)
    return {
        "n_audit_rows": int(len(audit)),
        "explained_variance": explained,
        "assessment_counts": audit["outlier_assessment"].value_counts().to_dict(),
        "outputs": {
            "audit": str(AUDIT_OUTPUT.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PCA outliers in final compact5 k=2 clustering.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        result = run_audit()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Final PCA outlier audit complete")
    print(f"- audit rows: {result['n_audit_rows']}")
    print(f"- explained variance: {result['explained_variance']}")
    print(f"- assessment counts: {result['assessment_counts']}")
    print(f"- outputs: {result['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
