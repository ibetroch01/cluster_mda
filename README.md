# AWV Cycling Count Clustering

Small data science project that clusters AWV bicycle counting locations in
Flanders using relative temporal cycling patterns from 2023-2025.

The final model uses:

- `FIETSERS` counts only
- location groups based on sites within 100 metres
- compact5 temporal features
- `StandardScaler`
- K-means with `k = 2`

The full methodological explanation is in the report/notebook, not in this
README.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Place the raw AWV files locally in:

```text
data/raw/
```

The `data/`, `.venv/`, `.vscode/`, and `reports/` folders are ignored by Git.

## Project Structure

```text
src/          analysis scripts
dashboard/    Shiny dashboard
outputs/      generated figures, tables, and models
data/         local raw and processed data, not tracked by Git
```

## Run The Workflow

Run everything:

```bash
make all
```

Or run separate stages:

```bash
make explore
make clean-data
make features
make model-building
make model-selection
make results
make dashboard-assets
```

## Dashboard

```bash
make dashboard
```

Open:

```text
http://127.0.0.1:8001
```

## Main Outputs

- final clusters: `data/processed/final_location_group_clusters_k2_compact5.csv`
- final cluster summary: `outputs/final_cluster_summary_k2_compact5.csv`
- final figures and maps: `outputs/`
- dashboard app: `dashboard/app.py`
