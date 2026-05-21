"""Create interactive Leaflet maps for final k=2 and exploratory k=3 clusters."""

import argparse
import sys
from pathlib import Path

import folium
import pandas as pd
from branca.element import MacroElement, Template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
K2_INPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
K3_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_clusters_k3_compact5.csv"
K2_OUTPUT = PROJECT_ROOT / "outputs" / "final_k2_cluster_map_leaflet.html"
K3_OUTPUT = PROJECT_ROOT / "outputs" / "exploratory_k3_cluster_map_leaflet.html"

COLORS = {
    "broad mixed / regular-use": "#4C78A8",
    "strongly seasonal / recreational-like": "#F58518",
    "mixed daytime / regular-use": "#72B7B2",
    "commuter-like / peak-oriented regular-use": "#54A24B",
}


def add_legend(map_object: folium.Map, title: str, labels: list[tuple[str, int]]) -> None:
    """Add a fixed legend to a Folium map."""
    rows = "\n".join(
        f"""
        <div style="display:flex; align-items:center; margin:4px 0;">
          <span style="background:{COLORS[label]}; width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:8px;"></span>
          <span>{label} (n={n})</span>
        </div>
        """
        for label, n in labels
    )
    html = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background: rgba(255,255,255,0.94);
        padding: 12px 14px;
        border: 1px solid #cfcfcf;
        border-radius: 4px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-width: 320px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.18);
    ">
      <div style="font-weight:700; margin-bottom:6px;">{title}</div>
      {rows}
      <div style="margin-top:8px; color:#555; line-height:1.3;">
        Coordinates are shown for interpretation only; they were not used as clustering features.
      </div>
    </div>
    {{% endmacro %}}
    """
    legend = MacroElement()
    legend._template = Template(html)
    map_object.get_root().add_child(legend)


def base_map(data: pd.DataFrame, title: str) -> folium.Map:
    """Create a Folium base map centered on the data."""
    center = [float(data["mean_latitude"].mean()), float(data["mean_longitude"].mean())]
    map_object = folium.Map(
        location=center,
        zoom_start=8,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(map_object)
    folium.TileLayer("CartoDB positron", name="CartoDB positron").add_to(map_object)
    title_html = f"""
    <div style="
        position: fixed;
        top: 16px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background: rgba(255,255,255,0.93);
        padding: 8px 14px;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        font-family: Arial, sans-serif;
        font-weight: 700;
        box-shadow: 0 1px 8px rgba(0,0,0,0.14);
    ">{title}</div>
    """
    map_object.get_root().html.add_child(folium.Element(title_html))
    return map_object


def label_k2(data: pd.DataFrame) -> pd.DataFrame:
    """Attach final k=2 cautious labels."""
    output = data.copy()
    seasonal_cluster = int(output.groupby("cluster_id")["log_summer_winter_ratio"].mean().idxmax())
    output["cluster_label"] = output["cluster_id"].map(
        lambda cluster_id: "strongly seasonal / recreational-like"
        if int(cluster_id) == seasonal_cluster
        else "broad mixed / regular-use"
    )
    return output


def label_k3(data: pd.DataFrame) -> pd.DataFrame:
    """Attach exploratory k=3 cautious labels."""
    output = data.copy()
    seasonal_cluster = int(output.groupby("k3_cluster_id")["log_summer_winter_ratio"].mean().idxmax())
    remaining = output.loc[output["k3_cluster_id"] != seasonal_cluster]
    commuter_cluster = int(remaining.groupby("k3_cluster_id")["weekday_commute_peak_share"].mean().idxmax())

    def label(cluster_id: int) -> str:
        if int(cluster_id) == seasonal_cluster:
            return "strongly seasonal / recreational-like"
        if int(cluster_id) == commuter_cluster:
            return "commuter-like / peak-oriented regular-use"
        return "mixed daytime / regular-use"

    output["cluster_label"] = output["k3_cluster_id"].map(label)
    return output


def marker_radius(label: str) -> int:
    """Use a slightly larger marker for the small seasonal cluster."""
    return 7 if label == "strongly seasonal / recreational-like" else 5


def add_points(map_object: folium.Map, data: pd.DataFrame, cluster_id_column: str) -> None:
    """Add cluster points to a Folium map."""
    for row in data.itertuples(index=False):
        label = row.cluster_label
        popup = folium.Popup(
            f"""
            <b>{row.location_group_id}</b><br>
            Gemeente: {getattr(row, "gemeente_values", "")}<br>
            Site IDs: {getattr(row, "site_ids", "")}<br>
            Cluster ID: {getattr(row, cluster_id_column)}<br>
            Label: {label}<br>
            log summer/winter: {row.log_summer_winter_ratio:.3f}<br>
            log weekend/weekday: {row.log_weekend_weekday_ratio:.3f}<br>
            weekday commute peak share: {row.weekday_commute_peak_share:.3f}<br>
            weekday midday share: {row.weekday_midday_share:.3f}<br>
            weekend midday/afternoon share: {row.weekend_midday_afternoon_share:.3f}
            """,
            max_width=360,
        )
        folium.CircleMarker(
            location=[float(row.mean_latitude), float(row.mean_longitude)],
            radius=marker_radius(label),
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color=COLORS[label],
            fill_opacity=0.86,
            popup=popup,
            tooltip=f"{row.location_group_id}: {label}",
        ).add_to(map_object)


def write_map(data: pd.DataFrame, output: Path, title: str, cluster_id_column: str) -> None:
    """Write one Leaflet cluster map."""
    data = data.dropna(subset=["mean_latitude", "mean_longitude"]).copy()
    map_object = base_map(data, title)
    add_points(map_object, data, cluster_id_column)
    legend_rows = (
        data.groupby("cluster_label")["location_group_id"]
        .nunique()
        .sort_index()
        .reset_index()
    )
    add_legend(
        map_object,
        title,
        [(row.cluster_label, int(row.location_group_id)) for row in legend_rows.itertuples(index=False)],
    )
    folium.LayerControl(collapsed=True).add_to(map_object)
    output.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(str(output))


def run_maps() -> dict[str, str]:
    """Create both actual basemap HTML maps."""
    if not K2_INPUT.exists():
        raise FileNotFoundError(f"Missing input: {K2_INPUT}")
    if not K3_INPUT.exists():
        raise FileNotFoundError(f"Missing input: {K3_INPUT}")
    k2 = label_k2(pd.read_csv(K2_INPUT))
    k3 = label_k3(pd.read_csv(K3_INPUT))
    write_map(k2, K2_OUTPUT, "Final k=2 compact5 clusters", "cluster_id")
    write_map(k3, K3_OUTPUT, "Exploratory k=3 compact5 clusters", "k3_cluster_id")
    return {
        "k2": str(K2_OUTPUT.relative_to(PROJECT_ROOT)),
        "k3": str(K3_OUTPUT.relative_to(PROJECT_ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Leaflet cluster maps on actual basemap tiles.")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        outputs = run_maps()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Leaflet cluster maps complete")
    print(f"- k2: {outputs['k2']}")
    print(f"- k3: {outputs['k3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
