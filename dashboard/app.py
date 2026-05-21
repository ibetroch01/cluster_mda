from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from shiny import App, reactive, render, ui


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

K2_CLUSTERS = DATA_DIR / "final_location_group_clusters_k2_compact5.csv"
K3_CLUSTERS = DATA_DIR / "location_group_clusters_k3_compact5.csv"
COUNTS = DATA_DIR / "counts_location_group_2023_2025.parquet"
DAY_QUALITY = DATA_DIR / "location_group_day_quality.csv"
QUALITY_SUMMARY = DATA_DIR / "location_group_quality_summary.csv"
FINAL_SUMMARY = OUTPUTS_DIR / "final_cluster_summary_k2_compact5.csv"
K3_SUMMARY = OUTPUTS_DIR / "k3_compact5_cluster_summary.csv"

COMPACT5 = [
    "log_weekend_weekday_ratio",
    "weekday_commute_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "log_summer_winter_ratio",
]

FEATURE_LABELS = {
    "log_weekend_weekday_ratio": "Weekend / weekday ratio",
    "weekday_commute_peak_share": "Weekday commute peak share",
    "weekday_midday_share": "Weekday midday share",
    "weekend_midday_afternoon_share": "Weekend midday-afternoon share",
    "log_summer_winter_ratio": "Summer / winter ratio",
}

DAY_NAMES = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

K2_LABELS = {
    0: "broad mixed / regular-use",
    1: "strongly seasonal / recreational-like",
}

K3_LABELS = {
    0: "mixed daytime / regular-use",
    1: "strongly seasonal / recreational-like",
    2: "commuter-like / peak-oriented regular-use",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _required_files() -> list[Path]:
    return [
        K2_CLUSTERS,
        K3_CLUSTERS,
        COUNTS,
        DAY_QUALITY,
        QUALITY_SUMMARY,
        FINAL_SUMMARY,
        K3_SUMMARY,
    ]


def _standardize_k2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = "k=2 final"
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce").astype("Int64")
    out["cluster_label"] = out["cluster_id"].map(K2_LABELS).fillna(out["cluster_name"])
    out = out.rename(
        columns={
            "distance_to_assigned_centroid": "distance_to_centroid",
            "mean_longitude": "longitude",
            "mean_latitude": "latitude",
        }
    )
    return out


def _standardize_k3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = "k=3 exploratory"
    out["cluster_id"] = pd.to_numeric(out["k3_cluster_id"], errors="coerce").astype("Int64")
    out["cluster_name"] = out["k3_cluster_name"]
    out["cluster_label"] = out["cluster_id"].map(K3_LABELS).fillna(out["k3_cluster_name"])
    out = out.rename(
        columns={
            "k3_silhouette_value": "silhouette_value",
            "k3_distance_to_centroid": "distance_to_centroid",
            "mean_longitude": "longitude",
            "mean_latitude": "latitude",
        }
    )
    return out


def _model_df(data, model: str) -> pd.DataFrame:
    return data.k3 if model == "k3" else data.k2


def _load_profiles(k2: pd.DataFrame, k3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_cols = ["location_group_id", "date", "count", "hour", "month", "day_of_week", "is_weekend"]
    counts = pd.read_parquet(COUNTS, columns=count_cols)
    counts["date"] = pd.to_datetime(counts["date"]).dt.date

    quality = _read_csv(DAY_QUALITY)
    quality["date"] = pd.to_datetime(quality["date"]).dt.date
    valid_days = quality.loc[
        (quality["valid_day"]) & (quality["daily_total"] > 0),
        ["location_group_id", "date", "daily_total", "day_of_week", "is_weekend", "month"],
    ].copy()

    counts = counts.merge(
        valid_days[["location_group_id", "date", "daily_total"]],
        on=["location_group_id", "date"],
        how="inner",
    )

    hourly_day = (
        counts.groupby(["location_group_id", "date", "is_weekend", "hour"], as_index=False)
        .agg(hourly_count=("count", "sum"), daily_total=("daily_total", "first"))
    )
    hourly_day["profile_value"] = hourly_day["hourly_count"] / hourly_day["daily_total"]
    hourly_day["day_type"] = np.where(hourly_day["is_weekend"], "weekend", "weekday")
    hourly_profile = (
        hourly_day.groupby(["location_group_id", "day_type", "hour"], as_index=False)
        .agg(profile_value=("profile_value", "mean"))
        .sort_values(["location_group_id", "day_type", "hour"])
    )

    group_mean = valid_days.groupby("location_group_id")["daily_total"].transform("mean")
    valid_days["profile_value"] = valid_days["daily_total"] / group_mean.replace(0, np.nan)

    weekday_profile = (
        valid_days.groupby(["location_group_id", "day_of_week"], as_index=False)
        .agg(profile_value=("profile_value", "mean"), mean_daily_total=("daily_total", "mean"))
        .sort_values(["location_group_id", "day_of_week"])
    )
    weekday_profile["weekday_name"] = weekday_profile["day_of_week"].map(DAY_NAMES)

    monthly_profile = (
        valid_days.groupby(["location_group_id", "month"], as_index=False)
        .agg(profile_value=("profile_value", "mean"), mean_daily_total=("daily_total", "mean"))
        .sort_values(["location_group_id", "month"])
    )
    monthly_profile["month_name"] = monthly_profile["month"].map(MONTH_NAMES)

    return hourly_profile, weekday_profile, monthly_profile


def _cluster_expectations(
    model_df: pd.DataFrame,
    hourly_profile: pd.DataFrame,
    weekday_profile: pd.DataFrame,
    monthly_profile: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    assignments = model_df[["location_group_id", "cluster_label"]].copy()

    hourly = (
        hourly_profile.merge(assignments, on="location_group_id", how="inner")
        .groupby(["cluster_label", "day_type", "hour"], as_index=False)
        .agg(cluster_profile_value=("profile_value", "mean"))
    )
    weekday = (
        weekday_profile.merge(assignments, on="location_group_id", how="inner")
        .groupby(["cluster_label", "day_of_week", "weekday_name"], as_index=False)
        .agg(cluster_profile_value=("profile_value", "mean"))
    )
    monthly = (
        monthly_profile.merge(assignments, on="location_group_id", how="inner")
        .groupby(["cluster_label", "month", "month_name"], as_index=False)
        .agg(cluster_profile_value=("profile_value", "mean"))
    )
    return {"hourly": hourly, "weekday": weekday, "monthly": monthly}


def _load_data():
    # Keep the dashboard simple: it only reads prepared outputs from the pipeline.
    missing = tuple(str(path.relative_to(PROJECT_ROOT)) for path in _required_files() if not path.exists())
    if missing:
        empty = pd.DataFrame()
        return SimpleNamespace(
            ready=False,
            missing_files=missing,
            k2=empty,
            k3=empty,
            quality=empty,
            hourly_profile=empty,
            weekday_profile=empty,
            monthly_profile=empty,
            cluster_profiles={},
            summaries={},
        )

    k2 = _standardize_k2(_read_csv(K2_CLUSTERS))
    k3 = _standardize_k3(_read_csv(K3_CLUSTERS))
    quality = _read_csv(QUALITY_SUMMARY)

    for df in (k2, k3):
        for feature in COMPACT5:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df["n_sites_in_group"] = pd.to_numeric(df["n_sites_in_group"], errors="coerce")
        df["silhouette_value"] = pd.to_numeric(df["silhouette_value"], errors="coerce")
        df["distance_to_centroid"] = pd.to_numeric(df["distance_to_centroid"], errors="coerce")

    hourly_profile, weekday_profile, monthly_profile = _load_profiles(k2, k3)
    cluster_profiles = {
        "k2": _cluster_expectations(k2, hourly_profile, weekday_profile, monthly_profile),
        "k3": _cluster_expectations(k3, hourly_profile, weekday_profile, monthly_profile),
    }

    summaries = {
        "k2": _read_csv(FINAL_SUMMARY),
        "k3": _read_csv(K3_SUMMARY),
    }
    return SimpleNamespace(
        ready=True,
        missing_files=missing,
        k2=k2,
        k3=k3,
        quality=quality,
        hourly_profile=hourly_profile,
        weekday_profile=weekday_profile,
        monthly_profile=monthly_profile,
        cluster_profiles=cluster_profiles,
        summaries=summaries,
    )


DATA = _load_data()


def _fig_html(fig: go.Figure, height: int = 520) -> ui.HTML:
    fig.update_layout(
        height=height,
        margin={"l": 40, "r": 24, "t": 48, "b": 108},
        legend_title_text="",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
        },
        font={"family": "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"},
    )
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )
    return ui.HTML(html)


def _empty_fig(message: str) -> ui.HTML:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _fig_html(fig, height=360)


def _ready_message() -> ui.Tag:
    if DATA.ready:
        return ui.div()
    return ui.div(
        ui.h3("Dashboard data not found"),
        ui.p("Run the preprocessing and final clustering scripts first."),
        ui.tags.ul(*(ui.tags.li(path) for path in DATA.missing_files)),
        class_="missing-data",
    )


def _location_choices() -> dict[str, str]:
    if not DATA.ready:
        return {}
    df = DATA.k2.sort_values(["gemeente_values", "location_group_id"])
    return {
        row.location_group_id: f"{row.location_group_id} | {row.gemeente_values} | sites {row.site_ids}"
        for row in df.itertuples()
    }


def _cluster_filter_choices(model: str) -> dict[str, str]:
    if not DATA.ready:
        return {"all": "All clusters"}
    labels = sorted(_model_df(DATA, model)["cluster_label"].dropna().unique())
    return {"all": "All clusters"} | {label: label for label in labels}


def _selected_location_row(location_group_id: str, model: str) -> pd.Series | None:
    if not DATA.ready:
        return None
    df = _model_df(DATA, model)
    rows = df.loc[df["location_group_id"] == location_group_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def _profile_plot(
    profile_df: pd.DataFrame,
    expectation_df: pd.DataFrame,
    location_group_id: str,
    cluster_label: str,
    x_col: str,
    y_title: str,
    title: str,
    x_label_map: dict[int, str] | None = None,
    color_col: str | None = None,
    height: int = 410,
) -> ui.HTML:
    own = profile_df.loc[profile_df["location_group_id"] == location_group_id].copy()
    expected = expectation_df.loc[expectation_df["cluster_label"] == cluster_label].copy()
    if own.empty or expected.empty:
        return _empty_fig("No profile data available for this location group.")

    own["series"] = "selected counter"
    expected["series"] = "cluster expectation"
    expected = expected.rename(columns={"cluster_profile_value": "profile_value"})
    plot_df = pd.concat([own, expected], ignore_index=True, sort=False)
    plot_df["line_dash"] = np.where(plot_df["series"].eq("cluster expectation"), "dash", "solid")

    if x_label_map:
        plot_df["x_label"] = plot_df[x_col].map(x_label_map)
        x_axis = "x_label"
        category_orders = {"x_label": [x_label_map[key] for key in sorted(x_label_map)]}
    else:
        x_axis = x_col
        category_orders = None

    fig = px.line(
        plot_df,
        x=x_axis,
        y="profile_value",
        color=color_col or "series",
        line_dash="series",
        category_orders=category_orders,
        markers=True,
        title=title,
        labels={"profile_value": y_title, x_axis: ""},
    )
    fig.update_traces(line={"width": 2.8})
    fig.update_yaxes(tickformat=".1%")
    return _fig_html(fig, height=height)


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Map",
        _ready_message(),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "map_model",
                    "Clustering model",
                    choices={"k2": "Final k=2 model", "k3": "Exploratory k=3 model"},
                ),
                ui.input_select("map_cluster", "Cluster filter", choices={"all": "All clusters"}),
                ui.p(
                    "The map is for interpretation only. Coordinates were not used as K-means features.",
                    class_="hint",
                ),
                width=310,
            ),
            ui.card(
                ui.card_header("Interactive location-group map"),
                ui.output_ui("map_plot"),
            ),
        ),
    ),
    ui.nav_panel(
        "Teller overview",
        _ready_message(),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "profile_model",
                    "Cluster expectation",
                    choices={"k2": "Final k=2 model", "k3": "Exploratory k=3 model"},
                ),
                ui.input_selectize("location_group", "Teller / location group", choices=_location_choices()),
                ui.p(
                    "Solid line = selected counter. Dotted line = average relative profile of the selected counter's cluster.",
                    class_="hint",
                ),
                width=360,
            ),
            ui.output_ui("location_summary"),
            ui.layout_columns(
                ui.card(ui.card_header("Hourly profile"), ui.output_ui("hourly_plot")),
                ui.card(ui.card_header("Daily profile"), ui.output_ui("daily_plot")),
                ui.card(ui.card_header("Monthly profile"), ui.output_ui("monthly_plot")),
                col_widths=[4, 4, 4],
            ),
        ),
    ),
    ui.nav_panel(
        "Feature explorer",
        _ready_message(),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select(
                    "feature_model",
                    "Model",
                    choices={"k2": "Final k=2 model", "k3": "Exploratory k=3 model"},
                ),
                ui.input_select(
                    "x_feature",
                    "X-axis",
                    choices={feature: FEATURE_LABELS[feature] for feature in COMPACT5},
                    selected="log_weekend_weekday_ratio",
                ),
                ui.input_select(
                    "y_feature",
                    "Y-axis",
                    choices={feature: FEATURE_LABELS[feature] for feature in COMPACT5},
                    selected="log_summer_winter_ratio",
                ),
                ui.p(
                    "This view is useful for seeing which temporal features separate the clusters. Volume, coordinates and quality variables are not clustering features.",
                    class_="hint",
                ),
                width=330,
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Interactive feature space"), ui.output_ui("feature_scatter")),
                ui.card(ui.card_header("Cluster feature profile"), ui.output_ui("feature_profile")),
                col_widths=[7, 5],
            ),
            ui.card(
                ui.card_header("Cluster summary"),
                ui.output_table("cluster_summary_table"),
            ),
        ),
    ),
    title="AWV cycling clustering dashboard",
    header=ui.tags.head(
        ui.tags.script(ui.HTML(get_plotlyjs())),
        ui.tags.style(
            """
            body { background: #f6f7f9; }
            .navbar { border-bottom: 1px solid #d9dee7; }
            .card { border-radius: 8px; border-color: #dde3ea; }
            .hint { color: #5f6b7a; font-size: 0.92rem; line-height: 1.35; }
            .missing-data {
                border: 1px solid #d6a63b;
                background: #fff7df;
                padding: 14px 16px;
                border-radius: 8px;
                margin-bottom: 14px;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-bottom: 14px;
            }
            .metric {
                background: white;
                border: 1px solid #dde3ea;
                border-radius: 8px;
                padding: 12px 14px;
            }
            .metric-label {
                color: #5f6b7a;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.02em;
            }
            .metric-value {
                color: #1f2937;
                font-size: 1.05rem;
                font-weight: 700;
                margin-top: 3px;
            }
            @media (max-width: 900px) {
                .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            """
        ),
    ),
)


def server(input, output, session):
    @reactive.effect
    def _update_map_cluster_filter() -> None:
        if DATA.ready:
            ui.update_select("map_cluster", choices=_cluster_filter_choices(input.map_model()))

    @render.ui
    def map_plot():
        if not DATA.ready:
            return _empty_fig("Run the preprocessing and final clustering scripts first.")

        df = _model_df(DATA, input.map_model()).copy()
        selected_cluster = input.map_cluster()
        if selected_cluster and selected_cluster != "all":
            df = df.loc[df["cluster_label"] == selected_cluster]

        fig = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color="cluster_label",
            hover_name="location_group_id",
            hover_data={
                "cluster_label": True,
                "gemeente_values": True,
                "site_ids": True,
                "n_sites_in_group": True,
                "silhouette_value": ":.3f",
                "distance_to_centroid": ":.3f",
                "latitude": ":.5f",
                "longitude": ":.5f",
            },
            zoom=7,
            height=650,
            title=f"Location groups colored by {input.map_model().replace('k', 'k=')}",
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox={"center": {"lat": 51.0, "lon": 4.5}},
        )
        fig.update_traces(marker={"size": 13, "opacity": 0.85})
        return _fig_html(fig, height=670)

    @render.ui
    def location_summary():
        if not DATA.ready:
            return ui.div()
        row = _selected_location_row(input.location_group(), input.profile_model())
        if row is None:
            return ui.div("No location group selected.")

        quality = DATA.quality.loc[DATA.quality["location_group_id"] == row["location_group_id"]]
        quality_row = quality.iloc[0] if not quality.empty else pd.Series(dtype=object)

        def metric(label: str, value: object) -> ui.Tag:
            return ui.div(
                ui.div(label, class_="metric-label"),
                ui.div(str(value), class_="metric-value"),
                class_="metric",
            )

        return ui.div(
            ui.div(
                metric("Location group", row["location_group_id"]),
                metric("Cluster", row["cluster_label"]),
                metric("Municipality", row.get("gemeente_values", "")),
                metric("Sites", row.get("site_ids", "")),
                metric("Valid days", int(quality_row.get("n_valid_days", 0) or 0)),
                metric(
                    "Full coverage",
                    f"{float(quality_row.get('percentage_full_coverage_intervals', np.nan)):.1%}"
                    if "percentage_full_coverage_intervals" in quality_row
                    else "n/a",
                ),
                metric("Silhouette", f"{float(row['silhouette_value']):.3f}"),
                metric("Centroid distance", f"{float(row['distance_to_centroid']):.3f}"),
                class_="metric-grid",
            )
        )

    @render.ui
    def hourly_plot():
        if not DATA.ready:
            return _empty_fig("No data loaded.")
        row = _selected_location_row(input.location_group(), input.profile_model())
        if row is None:
            return _empty_fig("No location group selected.")
        return _profile_plot(
            DATA.hourly_profile,
            DATA.cluster_profiles[input.profile_model()]["hourly"],
            row["location_group_id"],
            row["cluster_label"],
            x_col="hour",
            y_title="Average share of daily count",
            title="Hourly relative profile",
            color_col="day_type",
            height=430,
        )

    @render.ui
    def daily_plot():
        if not DATA.ready:
            return _empty_fig("No data loaded.")
        row = _selected_location_row(input.location_group(), input.profile_model())
        if row is None:
            return _empty_fig("No location group selected.")
        return _profile_plot(
            DATA.weekday_profile,
            DATA.cluster_profiles[input.profile_model()]["weekday"],
            row["location_group_id"],
            row["cluster_label"],
            x_col="day_of_week",
            y_title="Relative daily total",
            title="Day-of-week profile",
            x_label_map=DAY_NAMES,
            height=430,
        )

    @render.ui
    def monthly_plot():
        if not DATA.ready:
            return _empty_fig("No data loaded.")
        row = _selected_location_row(input.location_group(), input.profile_model())
        if row is None:
            return _empty_fig("No location group selected.")
        return _profile_plot(
            DATA.monthly_profile,
            DATA.cluster_profiles[input.profile_model()]["monthly"],
            row["location_group_id"],
            row["cluster_label"],
            x_col="month",
            y_title="Relative mean daily total",
            title="Monthly seasonal profile",
            x_label_map=MONTH_NAMES,
            height=430,
        )

    @render.ui
    def feature_scatter():
        if not DATA.ready:
            return _empty_fig("No data loaded.")
        df = _model_df(DATA, input.feature_model()).copy()
        x_feature = input.x_feature()
        y_feature = input.y_feature()
        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color="cluster_label",
            hover_name="location_group_id",
            hover_data={
                "gemeente_values": True,
                "site_ids": True,
                "silhouette_value": ":.3f",
                "distance_to_centroid": ":.3f",
            },
            title="Temporal feature space",
            labels={x_feature: FEATURE_LABELS[x_feature], y_feature: FEATURE_LABELS[y_feature]},
        )
        fig.update_traces(marker={"size": 11, "opacity": 0.82, "line": {"width": 0.5, "color": "#ffffff"}})
        return _fig_html(fig, height=560)

    @render.ui
    def feature_profile():
        if not DATA.ready:
            return _empty_fig("No data loaded.")
        df = _model_df(DATA, input.feature_model()).copy()
        z_cols = [f"z_{feature}" for feature in COMPACT5]
        profile = df.groupby("cluster_label", as_index=False)[z_cols].mean()
        long = profile.melt("cluster_label", var_name="feature", value_name="mean_z")
        long["feature"] = long["feature"].str.removeprefix("z_").map(FEATURE_LABELS)
        fig = px.bar(
            long,
            x="feature",
            y="mean_z",
            color="cluster_label",
            barmode="group",
            title="Mean standardized compact5 feature values",
            labels={"mean_z": "Mean z-score", "feature": ""},
        )
        fig.add_hline(y=0, line_color="#334155", line_width=1)
        fig.update_xaxes(tickangle=25)
        return _fig_html(fig, height=560)

    @render.table
    def cluster_summary_table():
        if not DATA.ready:
            return pd.DataFrame()
        df = _model_df(DATA, input.feature_model()).copy()
        summary = (
            df.groupby(["cluster_id", "cluster_label"], as_index=False)
            .agg(
                n=("location_group_id", "count"),
                mean_silhouette=("silhouette_value", "mean"),
                mean_distance=("distance_to_centroid", "mean"),
                mean_weekend_weekday=("log_weekend_weekday_ratio", "mean"),
                mean_commute_peak=("weekday_commute_peak_share", "mean"),
                mean_summer_winter=("log_summer_winter_ratio", "mean"),
            )
            .sort_values("cluster_id")
        )
        summary["pct"] = summary["n"] / summary["n"].sum()
        for col in [
            "mean_silhouette",
            "mean_distance",
            "mean_weekend_weekday",
            "mean_commute_peak",
            "mean_summer_winter",
            "pct",
        ]:
            summary[col] = summary[col].map(lambda x: f"{x:.3f}")
        return summary


app = App(app_ui, server)
