# AWV Cycling Count Clustering

This project analyses bicycle count data from AWV in Flanders for the period
2023-2025. The goal is to identify location groups with similar temporal cycling
patterns using unsupervised clustering.

The analysis is intentionally based on relative time-use patterns. Absolute
cycling volume, coordinates, municipality names, site identifiers, and data
quality variables are used for filtering, diagnostics, mapping, and
interpretation only. They are not used as K-means clustering features.

## Research Question

Can AWV bicycle counting locations in Flanders be grouped into interpretable
temporal usage profiles based on their relative daily, weekly, and seasonal
cycling patterns?

The resulting clusters should be read as temporal count patterns, not as direct
evidence of cyclist motivation. Terms such as recreational-like, commuter-like,
mixed, and regular-use are cautious interpretations of observed count profiles.

## Project Structure

```text
.
├── dashboard/          # Shiny dashboard for interactive exploration
├── data/
│   ├── raw/            # Local raw AWV CSV files, ignored by Git
│   └── processed/      # Generated processed datasets
├── outputs/            # Generated figures, tables, models, and diagnostics
├── src/                # Reproducible analysis scripts
├── Makefile            # Workflow commands
├── pyproject.toml      # Python dependencies and project metadata
└── README.md
```

The `data/` folder is intentionally excluded from version control. Raw AWV files
should be kept locally in `data/raw/`.

## 1. Data Exploration

The first step checks the raw AWV files before making assumptions about column
names or intervals.

- `src/inspect_data.py` lists available CSV files, normalizes detected columns,
  reports count-file date ranges, checks count types, and writes a schema summary.
- `src/diagnose_intervals.py` inspects interval lengths for cyclist counts and
  verifies that the observed 75-minute intervals are daylight-saving spring
  transition rows.

These scripts are diagnostic only. They do not create clustering features.

```bash
make explore
```

## 2. Data Cleaning

The cleaning stage creates an auditable quarter-hour-style site time series and
then aggregates nearby sites into location groups.

- `src/preprocess_counts.py` filters to `FIETSERS` rows from 2023-2025, parses
  timestamps, keeps normal 15-minute rows and valid 75-minute spring DST rows,
  removes negative counts, aggregates over directions, and saves a clean
  site-level time series.
- `src/group_locations.py` groups AWV sites within 100 metres using projected
  coordinates, then sums their counts at location-group level.
- `src/build_location_group_quality.py` computes valid-day and coverage
  diagnostics and creates the candidate eligible location groups.

Sites are grouped before feature engineering because nearby counters can
represent the same cycling location, for example opposite sides of the same
road. Missing intervals are not imputed as zero.

```bash
make clean-data
```

## 3. Feature Engineering

Feature engineering is performed at location-group level and uses valid days
only. Shares are calculated per day first and then averaged, so high-volume days
do not dominate the features.

- `src/engineer_location_group_features.py` creates candidate temporal features:
  weekend-weekday balance, weekday peak shares, weekday midday share, weekend
  midday-afternoon share, peak concentration, seasonality, night share, and daily
  variability.
- `src/analyze_location_group_correlations.py` checks feature correlations and
  writes the selected feature list used for modelling.

The final compact feature set is:

1. `log_weekend_weekday_ratio`
2. `weekday_commute_peak_share`
3. `weekday_midday_share`
4. `weekend_midday_afternoon_share`
5. `log_summer_winter_ratio`

`weekday_commute_peak_share` combines morning and evening peak shares:

```text
weekday_morning_peak_share + weekday_evening_peak_share
```

This compact set was chosen because it is interpretable, avoids unnecessary
double-counting of correlated indicators, and excludes optional diagnostic
variables such as night share and daily variability from the final model.

```bash
make features
```

## 4. Model Building

Model building uses K-means clustering on standardized relative temporal
features.

- `src/run_kmeans_elbow.py` computes elbow and silhouette diagnostics.
- `src/run_feature_set_sensitivity.py` compares full, core, and compact feature
  sets.
- `src/fit_final_kmeans_compact5.py` fits the final model.
- `src/explore_k3_compact5.py` fits an exploratory `k = 3` model to inspect
  substructure inside the broad final cluster.

Final model:

- Period: 2023-2025
- Data: `FIETSERS` only
- Unit of analysis: location groups
- Feature set: `compact5`
- Scaling: `StandardScaler`
- Model: K-means with `k = 2`
- Random state: `42`

```bash
make model-building
```

`make model` is kept as a backwards-compatible alias.

## 5. Model Selection

Model selection compares the candidate clustering choices and documents why the
final model is `compact5` with `k = 2`.

The decision is based on:

- elbow diagnostics;
- silhouette scores;
- feature-set sensitivity analysis;
- comparison of `k = 2`, `k = 3`, and `k = 4`;
- inspection of the small strongly seasonal cluster;
- PCA and centroid-distance outlier checks.

The final `k = 2` model was selected because it gave the clearest and most
stable separation in the compact feature space. The compact5 `k = 2` silhouette
score is approximately `0.597`, compared with approximately `0.306` for compact5
`k = 3`. The exploratory `k = 3` model is therefore used only as supporting
interpretation.

The final result separates 89 eligible location groups into:

- `cluster_0`: 82 location groups with a broad mixed / regular-use temporal
  pattern.
- `cluster_1`: 7 location groups with a strongly seasonal / recreational-like
  temporal pattern.

The small strongly seasonal cluster is checked separately with quality and PCA
outlier audits. These checks support the interpretation that the group reflects
unusual but valid temporal profiles rather than obvious data-quality failures.

Model-selection audit scripts:

- `src/audit_final_small_cluster.py` audits the seven-location seasonal cluster.
- `src/audit_final_pca_outliers.py` audits observations with unusual PCA or
  centroid-distance behaviour.

```bash
make model-selection
```

## 6. Dashboarding Results

The results stage prepares figures, maps, and interactive outputs for presenting
the final model and the exploratory `k = 3` comparison.

Dashboard-result scripts:

- `src/create_final_cluster_visualizations.py` creates feature profile, hourly,
  weekend, monthly, PCA, and detail figures/tables.
- `src/create_final_cluster_maps.py` creates static cluster maps.
- `src/create_leaflet_cluster_maps.py` creates interactive Leaflet maps.

```bash
make dashboard-results
```

`make results` is kept as a backwards-compatible alias.

The dashboard is built with Shiny for Python and reads prepared outputs only. It
does not recompute the full analysis.

- `dashboard/app.py` provides an interactive map, counter overview profiles, and
  additional cluster exploration visuals.

```bash
make dashboard
```

`make app` is kept as a shorter alias for the same command.

## Extra: Version Control

The project is designed to be kept in Git without committing local data or
environment files.

- Source code, dashboard code, `README.md`, `Makefile`, and `pyproject.toml`
  belong in version control.
- Raw and processed data stay local under `data/` and are ignored by Git.
- The virtual environment `.venv/` is ignored by Git.
- Python cache folders, test caches, and generated package metadata are ignored.
- Generated outputs can be kept locally for the dashboard and presentation, but
  should be reviewed before committing because some outputs may be large.

## Setup

Create a local virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The `.venv/` folder is ignored by Git. Installing with `pip install -e .` uses
the dependencies listed in `pyproject.toml` and makes local project code
available in editable mode.

## Reproducing the Project

Run the full workflow:

```bash
make all
```

Or run stages separately:

```bash
make explore
make clean-data
make features
make model-building
make model-selection
make dashboard-results
make dashboard
```

The dashboard starts at:

```text
http://127.0.0.1:8001
```

## Methodological Notes

- Only `FIETSERS` rows are used.
- Direction-level counts are summed to site level.
- Sites within 100 metres are grouped before feature engineering.
- Location-group counts are aggregated before features are created.
- Missing intervals are not treated as zero.
- Quality variables are used for filtering and diagnostics only.
- Absolute volume is not a clustering feature.
- Coordinates are used for maps only, after clustering.
- The clusters describe temporal count patterns and do not prove individual
  cyclist motives.
