"""Create final static basemap cluster maps for k=2 and exploratory k=3."""

import argparse
import math
import ssl
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image

from plot_style import CLUSTER_COLORS, save_figure, setup_matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
K2_INPUT = PROJECT_ROOT / "data" / "processed" / "final_location_group_clusters_k2_compact5.csv"
K3_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_clusters_k3_compact5.csv"
LOCATION_GROUPS_INPUT = PROJECT_ROOT / "data" / "processed" / "location_groups.csv"
K2_MAP_OUTPUT = PROJECT_ROOT / "outputs" / "fig_final_k2_cluster_map.png"
K3_MAP_OUTPUT = PROJECT_ROOT / "outputs" / "fig_exploratory_k3_cluster_map.png"

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
    "broad mixed / regular-use": CLUSTER_COLORS["regular"],
    "strongly seasonal / recreational-like": CLUSTER_COLORS["seasonal"],
    "mixed daytime / regular-use": CLUSTER_COLORS["mixed"],
    "commuter-like / peak-oriented regular-use": CLUSTER_COLORS["commuter"],
}
TILE_SIZE = 256
TILE_ZOOM = 8
TILE_URL = "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
TILE_ATTRIBUTION = "Basemap: CartoDB Positron, OpenStreetMap contributors. Coordinates were not clustering features."
EARTH_RADIUS = 6378137.0
SSL_CONTEXT = ssl._create_unverified_context()


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


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Convert lon/lat to fractional slippy-map tile coordinates."""
    lat = min(max(lat, -85.05112878), 85.05112878)
    n_tiles = 2**zoom
    x_tile = (lon + 180.0) / 360.0 * n_tiles
    lat_rad = math.radians(lat)
    y_tile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0
    return x_tile, y_tile * n_tiles


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Convert lon/lat to Web Mercator metres for plotting on map tiles."""
    lat = min(max(lat, -85.05112878), 85.05112878)
    x = EARTH_RADIUS * math.radians(lon)
    y = EARTH_RADIUS * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def fetch_tile(x_tile: int, y_tile: int, zoom: int) -> Image.Image:
    """Download one CartoDB Positron map tile."""
    url = TILE_URL.format(z=zoom, x=x_tile, y=y_tile)
    request = Request(url, headers={"User-Agent": "awv-cycling-clustering-report/1.0"})
    try:
        with urlopen(request, timeout=20, context=SSL_CONTEXT) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not download basemap tiles for the static report map. "
            "Check the internet connection, or use src/create_leaflet_cluster_maps.py "
            "to create the interactive dashboard map instead."
        ) from exc


def padded_bounds(data: pd.DataFrame) -> tuple[float, float, float, float]:
    """Create a slightly padded lon/lat bounding box around all points."""
    lon_min = float(data["mean_longitude"].min())
    lon_max = float(data["mean_longitude"].max())
    lat_min = float(data["mean_latitude"].min())
    lat_max = float(data["mean_latitude"].max())
    lon_pad = max((lon_max - lon_min) * 0.08, 0.10)
    lat_pad = max((lat_max - lat_min) * 0.12, 0.06)
    return lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad


def basemap_image(bounds: tuple[float, float, float, float], zoom: int) -> Image.Image:
    """Build a cropped basemap image for a lon/lat bounding box."""
    lon_min, lat_min, lon_max, lat_max = bounds
    x0, y0 = lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = lonlat_to_tile(lon_max, lat_min, zoom)
    x_start, x_end = math.floor(x0), math.floor(x1)
    y_start, y_end = math.floor(y0), math.floor(y1)

    canvas = Image.new(
        "RGB",
        ((x_end - x_start + 1) * TILE_SIZE, (y_end - y_start + 1) * TILE_SIZE),
        "#f8fafc",
    )
    for x_tile in range(x_start, x_end + 1):
        for y_tile in range(y_start, y_end + 1):
            tile = fetch_tile(x_tile, y_tile, zoom)
            canvas.paste(tile, ((x_tile - x_start) * TILE_SIZE, (y_tile - y_start) * TILE_SIZE))

    crop_box = (
        int(round((x0 - x_start) * TILE_SIZE)),
        int(round((y0 - y_start) * TILE_SIZE)),
        int(round((x1 - x_start) * TILE_SIZE)),
        int(round((y1 - y_start) * TILE_SIZE)),
    )
    return canvas.crop(crop_box)


def set_lonlat_ticks(ax, bounds: tuple[float, float, float, float]) -> None:
    """Show readable longitude/latitude ticks on Web Mercator axes."""
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_ticks = np.linspace(math.ceil(lon_min * 2) / 2, math.floor(lon_max * 2) / 2, 5)
    lat_ticks = np.linspace(math.ceil(lat_min * 5) / 5, math.floor(lat_max * 5) / 5, 5)
    lon_ticks = [float(tick) for tick in lon_ticks if lon_min <= tick <= lon_max]
    lat_ticks = [float(tick) for tick in lat_ticks if lat_min <= tick <= lat_max]
    ax.set_xticks([lonlat_to_mercator(lon, lat_min)[0] for lon in lon_ticks])
    ax.set_xticklabels([f"{lon:.1f}" for lon in lon_ticks])
    ax.set_yticks([lonlat_to_mercator(lon_min, lat)[1] for lat in lat_ticks])
    ax.set_yticklabels([f"{lat:.1f}" for lat in lat_ticks])


def plot_map(
    data: pd.DataFrame,
    output: Path,
    title: str,
    cluster_id_column: str,
) -> pd.DataFrame:
    """Create a static basemap cluster map and return legend summary."""
    plt, _ = setup_matplotlib()
    bounds = padded_bounds(data)
    background = basemap_image(bounds, TILE_ZOOM)
    lon_min, lat_min, lon_max, lat_max = bounds
    x_min, y_min = lonlat_to_mercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_mercator(lon_max, lat_max)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.imshow(background, extent=[x_min, x_max, y_min, y_max], origin="upper", zorder=0)
    summary = (
        data.groupby(["cluster_label", cluster_id_column], as_index=False)
        .agg(n=("location_group_id", "nunique"))
        .sort_values("cluster_label")
    )
    for row in summary.itertuples(index=False):
        subset = data.loc[data["cluster_label"] == row.cluster_label]
        x_values, y_values = zip(
            *[lonlat_to_mercator(lon, lat) for lon, lat in zip(subset["mean_longitude"], subset["mean_latitude"], strict=True)]
        )
        marker_size = 58 if row.n < 15 else 44
        ax.scatter(
            x_values,
            y_values,
            s=marker_size,
            color=COLORS.get(row.cluster_label, "#777777"),
            edgecolor="white",
            linewidth=0.9,
            alpha=0.93,
            label=f"{row.cluster_label} (n={int(row.n)})",
            zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    set_lonlat_ticks(ax, bounds)
    ax.grid(color="#ffffff", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(frameon=True, loc="lower left", bbox_to_anchor=(0.015, 0.02), ncol=1, prop={"size": 8.5})
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_edgecolor("#d8dee6")
    ax.text(
        0.99,
        0.98,
        TILE_ATTRIBUTION,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#4A4A4A",
        bbox={"facecolor": "white", "edgecolor": "#d8dee6", "alpha": 0.82, "pad": 2.5},
    )
    fig.subplots_adjust(bottom=0.12)
    save_figure(fig, output)
    plt.close(fig)
    return summary


def run_maps() -> dict[str, object]:
    """Create both final static cluster maps."""
    k2, k3, groups = load_inputs()
    k2 = label_k2(ensure_coordinates(k2, groups))
    k3 = label_k3(ensure_coordinates(k3, groups))
    k2_summary = plot_map(k2, K2_MAP_OUTPUT, "Final k=2 compact5 clusters", "cluster_id")
    k3_summary = plot_map(k3, K3_MAP_OUTPUT, "Exploratory k=3 compact5 clusters", "k3_cluster_id")
    return {
        "k2_counts": k2_summary[["cluster_label", "n"]].to_dict(orient="records"),
        "k3_counts": k3_summary[["cluster_label", "n"]].to_dict(orient="records"),
        "outputs": {
            "k2_map": str(K2_MAP_OUTPUT.relative_to(PROJECT_ROOT)),
            "k3_map": str(K3_MAP_OUTPUT.relative_to(PROJECT_ROOT)),
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
