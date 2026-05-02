"""Create final static cluster maps for k=2 and exploratory k=3."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
K2_INPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
K3_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_clusters_k3_compact5.csv"
LOCATION_GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
K2_MAP_OUTPUT = PROJECT_ROOT / "outputs" / "fig_final_k2_cluster_map.png"
K3_MAP_OUTPUT = PROJECT_ROOT / "outputs" / "fig_exploratory_k3_cluster_map.png"
REPORT_OUTPUT = PROJECT_ROOT / "reports" / "map_interpretation.md"

K2_LABELS = {
    "broad": "broad mixed / regular-use",
    "seasonal": "strongly seasonal / recreational-like",
}
K3_LABELS = {
    "mixed": "mixed daytime / regular-use",
    "seasonal": "strongly seasonal / recreational-like",
    "commuter": "commuter-like / peak-oriented regular-use",
}
COLORS = {
    "broad mixed / regular-use": "#4C78A8",
    "strongly seasonal / recreational-like": "#F58518",
    "mixed daytime / regular-use": "#72B7B2",
    "commuter-like / peak-oriented regular-use": "#54A24B",
}


def setup_matplotlib():
    """Import matplotlib with a writable local cache."""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cluster and location metadata inputs."""
    for path in [K2_INPUT, K3_INPUT, LOCATION_GROUPS_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(K2_INPUT), pd.read_csv(K3_INPUT), pd.read_csv(LOCATION_GROUPS_INPUT)


def ensure_coordinates(clusters: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Attach mean longitude and latitude from location_groups.csv."""
    coordinate_columns = ["location_group_id", "mean_longitude", "mean_latitude"]
    output = clusters.drop(columns=[c for c in ["mean_longitude", "mean_latitude"] if c in clusters.columns])
    output = output.merge(groups[coordinate_columns], on="location_group_id", how="left")
    return output.dropna(subset=["mean_longitude", "mean_latitude"]).copy()


def label_k2(clusters: pd.DataFrame) -> pd.DataFrame:
    """Assign cautious k=2 map labels."""
    output = clusters.copy()
    seasonal_cluster = int(output.groupby("cluster_id")["log_summer_winter_ratio"].mean().idxmax())
    output["cluster_label"] = output["cluster_id"].map(
        lambda cluster_id: K2_LABELS["seasonal"] if int(cluster_id) == seasonal_cluster else K2_LABELS["broad"]
    )
    return output


def label_k3(clusters: pd.DataFrame) -> pd.DataFrame:
    """Assign cautious exploratory k=3 map labels from compact5 profiles."""
    output = clusters.copy()
    seasonal_cluster = int(output.groupby("k3_cluster_id")["log_summer_winter_ratio"].mean().idxmax())
    remaining = output.loc[output["k3_cluster_id"] != seasonal_cluster]
    commuter_cluster = int(remaining.groupby("k3_cluster_id")["weekday_commute_peak_share"].mean().idxmax())

    def label(cluster_id: int) -> str:
        if int(cluster_id) == seasonal_cluster:
            return K3_LABELS["seasonal"]
        if int(cluster_id) == commuter_cluster:
            return K3_LABELS["commuter"]
        return K3_LABELS["mixed"]

    output["cluster_label"] = output["k3_cluster_id"].map(label)
    return output


def plot_map(
    data: pd.DataFrame,
    output: Path,
    title: str,
    cluster_id_column: str,
) -> pd.DataFrame:
    """Create a static lon/lat cluster map and return legend summary."""
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 8.4))
    summary = (
        data.groupby(["cluster_label", cluster_id_column], as_index=False)
        .agg(n=("location_group_id", "nunique"))
        .sort_values("cluster_label")
    )
    for row in summary.itertuples(index=False):
        subset = data.loc[data["cluster_label"] == row.cluster_label]
        marker_size = 62 if row.n < 15 else 48
        ax.scatter(
            subset["mean_longitude"],
            subset["mean_latitude"],
            s=marker_size,
            color=COLORS.get(row.cluster_label, "#777777"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.86,
            label=f"{row.cluster_label} (n={int(row.n)})",
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.22)
    ax.legend(frameon=True, loc="best", fontsize=9)
    ax.text(
        0.01,
        0.01,
        "Coordinates are shown for interpretation only; they were not used as clustering features.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3},
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def write_report(k2_summary: pd.DataFrame, k3_summary: pd.DataFrame) -> None:
    """Write cautious map interpretation report."""
    lines = [
        "# Kaartinterpretatie AWV clustering",
        "",
        f"Gegenereerd op: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Belangrijk",
        "",
        "De kaarten gebruiken gemiddelde lengte- en breedtegraden alleen voor interpretatie achteraf. "
        "Coördinaten, gemeenten en site-identiteiten zijn niet gebruikt als clusteringfeatures. "
        "De clusters beschrijven relatieve temporele telpatronen en bewijzen geen fietsmotieven.",
        "",
        "## Finale k=2 kaart",
        "",
    ]
    for row in k2_summary.itertuples(index=False):
        lines.append(f"- `{row.cluster_label}`: {int(row.n)} locatiegroepen.")
    lines.extend(
        [
            "",
            "De finale k=2 kaart toont vooral een kleine strongly seasonal / recreational-like cluster tegenover een brede mixed / regular-use groep. "
            "Geografische patronen mogen alleen voorzichtig worden gelezen; de kaart is bedoeld als inspectie, niet als modelinput.",
            "",
            "## Exploratieve k=3 kaart",
            "",
        ]
    )
    for row in k3_summary.itertuples(index=False):
        lines.append(f"- `{row.cluster_label}`: {int(row.n)} locatiegroepen.")
    lines.extend(
        [
            "",
            "De k=3 kaart is exploratief. Ze behoudt de strongly seasonal / recreational-like groep en splitst de brede groep in een meer mixed daytime / regular-use profiel en een meer commuter-like / peak-oriented regular-use profiel. "
            "Deze termen blijven voorzichtig en verwijzen naar temporele patronen, niet naar bewezen motieven.",
            "",
        ]
    )
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def run_maps() -> dict[str, object]:
    """Create both final static cluster maps and report."""
    k2, k3, groups = load_inputs()
    k2 = label_k2(ensure_coordinates(k2, groups))
    k3 = label_k3(ensure_coordinates(k3, groups))
    k2_summary = plot_map(k2, K2_MAP_OUTPUT, "Final k=2 compact5 clusters", "cluster_id")
    k3_summary = plot_map(k3, K3_MAP_OUTPUT, "Exploratory k=3 compact5 clusters", "k3_cluster_id")
    write_report(k2_summary, k3_summary)
    return {
        "k2_counts": k2_summary[["cluster_label", "n"]].to_dict(orient="records"),
        "k3_counts": k3_summary[["cluster_label", "n"]].to_dict(orient="records"),
        "outputs": {
            "k2_map": str(K2_MAP_OUTPUT.relative_to(PROJECT_ROOT)),
            "k3_map": str(K3_MAP_OUTPUT.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final static cluster maps.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        result = run_maps()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Final cluster maps complete")
    print(f"- k2 counts: {result['k2_counts']}")
    print(f"- k3 counts: {result['k3_counts']}")
    print(f"- outputs: {result['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
