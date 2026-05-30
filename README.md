# AWV Cycling Count Clustering

Small data science project that clusters AWV bicycle counting locations in
Flanders using relative temporal cycling patterns from 2023-2025.

The final model uses:

- `FIETSERS` counts only
- location groups based on sites within 100 metres
- compact5 temporal features
- `StandardScaler`
- K-means with `k = 2`


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
```

## Dashboard

```bash
make dashboard
```

Open:

```text
http://127.0.0.1:8001
```

