.PHONY: final-model final-figures k3-exploration app

PYTHON ?= python3

final-model:
	$(PYTHON) src/fit_final_kmeans_compact5.py

final-figures:
	$(PYTHON) src/create_final_cluster_visualizations.py
	$(PYTHON) src/create_final_cluster_maps.py
	$(PYTHON) src/create_leaflet_cluster_maps.py

k3-exploration:
	$(PYTHON) src/explore_k3_compact5.py

app:
	shiny run --host 127.0.0.1 --port 8001 dashboard/app.py
