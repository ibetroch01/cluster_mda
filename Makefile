.PHONY: all explore clean-data features model-building model-selection dashboard-results model results dashboard app final-model final-figures k3-exploration

PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
SHINY := $(shell if [ -x .venv/bin/shiny ]; then echo .venv/bin/shiny; else echo shiny; fi)

all: explore clean-data features model-building model-selection dashboard-results

explore:
	$(PYTHON) src/inspect_data.py
	$(PYTHON) src/diagnose_intervals.py

clean-data:
	$(PYTHON) src/preprocess_counts.py
	$(PYTHON) src/group_locations.py
	$(PYTHON) src/build_location_group_quality.py

features:
	$(PYTHON) src/engineer_location_group_features.py
	$(PYTHON) src/analyze_location_group_correlations.py

model-building:
	$(PYTHON) src/run_kmeans_elbow.py
	$(PYTHON) src/run_feature_set_sensitivity.py
	$(PYTHON) src/fit_final_kmeans_compact5.py
	$(PYTHON) src/explore_k3_compact5.py

model-selection:
	$(PYTHON) src/audit_final_small_cluster.py
	$(PYTHON) src/audit_final_pca_outliers.py

dashboard-results:
	$(PYTHON) src/create_final_cluster_visualizations.py
	$(PYTHON) src/create_final_cluster_maps.py
	$(PYTHON) src/create_leaflet_cluster_maps.py

dashboard:
	$(SHINY) run --host 127.0.0.1 --port 8001 dashboard/app.py

app: dashboard

model: model-building

results: dashboard-results

final-model:
	$(PYTHON) src/fit_final_kmeans_compact5.py

final-figures:
	$(PYTHON) src/create_final_cluster_visualizations.py
	$(PYTHON) src/create_final_cluster_maps.py
	$(PYTHON) src/create_leaflet_cluster_maps.py

k3-exploration:
	$(PYTHON) src/explore_k3_compact5.py
