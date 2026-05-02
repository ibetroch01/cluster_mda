# AWV Cycling Count Clustering

This project analyses AWV bicycle counter data for Flanders over 2023-2025.
The unit of analysis is a location group: counting sites within 100 metres are
grouped and their counts are aggregated before feature engineering.

The final model uses relative temporal patterns only. Absolute volume,
coordinates, municipality, site identifiers, and data-quality variables are used
for diagnostics and interpretation, not as K-means features.

## Final Model

- Data: `FIETSERS` only
- Period: 2023-2025
- Feature set: `compact5`
- Standardization: `StandardScaler`
- Model: K-means with `k = 2`
- Final result:
  - `cluster_0`: 82 location groups, broad mixed / regular-use
  - `cluster_1`: 7 location groups, strongly seasonal / recreational-like

The cluster labels are cautious descriptions of temporal count patterns. They do
not prove cyclist motives.

## Compact5 Features

1. `log_weekend_weekday_ratio`
2. `weekday_commute_peak_share`
3. `weekday_midday_share`
4. `weekend_midday_afternoon_share`
5. `log_summer_winter_ratio`

`weekday_commute_peak_share` is computed as:

```text
weekday_morning_peak_share + weekday_evening_peak_share
```

## Main Deliverables

- `data/processed/final_location_group_clusters_k2_compact5.csv`
- `data/processed/location_group_clusters_k3_compact5.csv`
- `outputs/final_cluster_summary_k2_compact5.csv`
- `outputs/feature_set_model_comparison.csv`
- `outputs/final_small_cluster_audit.csv`
- `outputs/final_pca_outlier_audit.csv`

## Reproducibility Commands

```bash
make final-model
make final-figures
make k3-exploration
```

## Notes

The exploratory `k = 3` solution is not the main model. It is used only to
inspect substructure inside the broad final `k = 2` cluster.

Geographic maps use coordinates only after clustering for interpretation.
Coordinates were not used as clustering features.
