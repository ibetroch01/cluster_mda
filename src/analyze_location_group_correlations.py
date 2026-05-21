"""Check correlations between candidate clustering features.

This is a feature-selection step, not a modelling step.  The output stays
unscaled so that scaling only happens inside the K-means scripts.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_candidate.csv"
DEFAULT_PEARSON_OUTPUT = PROJECT_ROOT / "outputs" / "location_group_correlation_pearson.csv"
DEFAULT_SPEARMAN_OUTPUT = PROJECT_ROOT / "outputs" / "location_group_correlation_spearman.csv"
DEFAULT_FLAGS_OUTPUT = PROJECT_ROOT / "outputs" / "location_group_correlation_flags.csv"
DEFAULT_SELECTED_JSON_OUTPUT = PROJECT_ROOT / "outputs" / "selected_features.json"
DEFAULT_SELECTED_TABLE_OUTPUT = PROJECT_ROOT / "data" / "processed" / "location_group_features_selected.csv"

CORRELATION_THRESHOLD = 0.85
NIGHT_SHARE_NEAR_ZERO_THRESHOLD = 0.01
NIGHT_SHARE_NEAR_ZERO_SHARE_THRESHOLD = 0.80

CORE_CANDIDATE_FEATURES = [
    "log_weekend_weekday_ratio",
    "weekday_morning_peak_share",
    "weekday_evening_peak_share",
    "weekday_midday_share",
    "weekend_midday_afternoon_share",
    "weekday_peak_concentration_2h",
    "log_summer_winter_ratio",
]
OPTIONAL_FEATURES = [
    "night_share",
    "daily_variability_cv",
]
ALL_CANDIDATE_FEATURES = CORE_CANDIDATE_FEATURES + OPTIONAL_FEATURES


def load_features(path: Path) -> pd.DataFrame:
    """Load and validate the candidate feature table."""
    if not path.exists():
        raise FileNotFoundError(f"Candidate feature table not found: {path}")
    features = pd.read_csv(path)
    required = {"location_group_id", *CORE_CANDIDATE_FEATURES}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Candidate feature table missing columns: {', '.join(missing)}")

    available_optional = [column for column in OPTIONAL_FEATURES if column in features.columns]
    for column in CORE_CANDIDATE_FEATURES + available_optional:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def filter_complete_core(features: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values in core candidate features."""
    clean = features.replace([np.inf, -np.inf], np.nan)
    return clean.dropna(subset=CORE_CANDIDATE_FEATURES).copy()


def calculate_correlations(features: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson and Spearman correlation matrices."""
    matrix = features[columns].copy()
    return matrix.corr(method="pearson"), matrix.corr(method="spearman")


def build_correlation_flags(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Return feature pairs with absolute Pearson or Spearman correlation above threshold."""
    records: list[dict[str, object]] = []
    columns = list(pearson.columns)
    for index, feature_1 in enumerate(columns):
        for feature_2 in columns[index + 1 :]:
            pearson_value = pearson.loc[feature_1, feature_2]
            spearman_value = spearman.loc[feature_1, feature_2]
            if pd.notna(pearson_value) and abs(float(pearson_value)) > threshold:
                records.append(
                    {
                        "method": "pearson",
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "correlation": float(pearson_value),
                        "abs_correlation": abs(float(pearson_value)),
                    }
                )
            if pd.notna(spearman_value) and abs(float(spearman_value)) > threshold:
                records.append(
                    {
                        "method": "spearman",
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "correlation": float(spearman_value),
                        "abs_correlation": abs(float(spearman_value)),
                    }
                )
    return pd.DataFrame.from_records(
        records,
        columns=["method", "feature_1", "feature_2", "correlation", "abs_correlation"],
    ).sort_values(["abs_correlation", "method"], ascending=[False, True], ignore_index=True)


def max_abs_correlation(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    feature_1: str,
    feature_2: str,
) -> float:
    """Return the maximum absolute Pearson/Spearman correlation for a pair."""
    values = []
    for matrix in [pearson, spearman]:
        if feature_1 in matrix.index and feature_2 in matrix.columns:
            value = matrix.loc[feature_1, feature_2]
            if pd.notna(value):
                values.append(abs(float(value)))
    return max(values) if values else float("nan")


def is_strongly_correlated(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    threshold: float = CORRELATION_THRESHOLD,
) -> bool:
    """Check whether either Pearson or Spearman absolute correlation is strong."""
    value = max_abs_correlation(pearson, spearman, feature_1, feature_2)
    return bool(pd.notna(value) and value > threshold)


def recommend_features(
    features: pd.DataFrame,
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Create composite features where useful and recommend an unscaled feature list."""
    output = features.copy()
    selected = [
        "log_weekend_weekday_ratio",
        "weekday_morning_peak_share",
        "weekday_evening_peak_share",
        "weekday_midday_share",
        "weekend_midday_afternoon_share",
        "weekday_peak_concentration_2h",
        "log_summer_winter_ratio",
    ]
    rationale: dict[str, object] = {
        "correlation_threshold": CORRELATION_THRESHOLD,
        "night_share_near_zero_threshold": NIGHT_SHARE_NEAR_ZERO_THRESHOLD,
        "night_share_near_zero_share_threshold": NIGHT_SHARE_NEAR_ZERO_SHARE_THRESHOLD,
        "decisions": [],
    }

    morning_evening_corr = max_abs_correlation(
        pearson,
        spearman,
        "weekday_morning_peak_share",
        "weekday_evening_peak_share",
    )
    if is_strongly_correlated(pearson, spearman, "weekday_morning_peak_share", "weekday_evening_peak_share"):
        output["weekday_commute_peak_share"] = (
            output["weekday_morning_peak_share"] + output["weekday_evening_peak_share"]
        )
        selected = [
            "weekday_commute_peak_share" if column == "weekday_morning_peak_share" else column
            for column in selected
            if column != "weekday_evening_peak_share"
        ]
        rationale["decisions"].append(
            {
                "decision": "created_weekday_commute_peak_share",
                "reason": "weekday_morning_peak_share and weekday_evening_peak_share are strongly correlated",
                "max_abs_correlation": morning_evening_corr,
            }
        )
    else:
        rationale["decisions"].append(
            {
                "decision": "kept_morning_and_evening_peak_shares",
                "reason": "weekday_morning_peak_share and weekday_evening_peak_share are not strongly correlated",
                "max_abs_correlation": morning_evening_corr,
            }
        )

    commute_features = (
        ["weekday_commute_peak_share"]
        if "weekday_commute_peak_share" in output.columns
        else ["weekday_morning_peak_share", "weekday_evening_peak_share"]
    )
    peak_concentration_correlations = {
        feature: max_abs_correlation(pearson, spearman, "weekday_peak_concentration_2h", feature)
        for feature in commute_features
        if feature in pearson.columns
    }
    if "weekday_commute_peak_share" in output.columns:
        temporary = output[["weekday_peak_concentration_2h", "weekday_commute_peak_share"]].corr()
        peak_concentration_correlations["weekday_commute_peak_share"] = abs(
            float(temporary.loc["weekday_peak_concentration_2h", "weekday_commute_peak_share"])
        )
    peak_concentration_is_redundant = any(
        pd.notna(value) and value > CORRELATION_THRESHOLD
        for value in peak_concentration_correlations.values()
    )
    if peak_concentration_is_redundant and "weekday_peak_concentration_2h" in selected:
        selected.remove("weekday_peak_concentration_2h")
        rationale["decisions"].append(
            {
                "decision": "dropped_weekday_peak_concentration_2h",
                "reason": "weekday_peak_concentration_2h is highly correlated with commute peak features",
                "max_abs_correlations": peak_concentration_correlations,
            }
        )
    else:
        rationale["decisions"].append(
            {
                "decision": "kept_weekday_peak_concentration_2h",
                "reason": "weekday_peak_concentration_2h is not highly correlated with commute peak features",
                "max_abs_correlations": peak_concentration_correlations,
            }
        )

    if "night_share" in output.columns:
        near_zero_share = float((output["night_share"].abs() <= NIGHT_SHARE_NEAR_ZERO_THRESHOLD).mean())
        if near_zero_share > NIGHT_SHARE_NEAR_ZERO_SHARE_THRESHOLD:
            rationale["decisions"].append(
                {
                    "decision": "dropped_night_share",
                    "reason": "more than 80% of night_share values are near zero",
                    "near_zero_share": near_zero_share,
                }
            )
        else:
            selected.append("night_share")
            rationale["decisions"].append(
                {
                    "decision": "kept_night_share",
                    "reason": "night_share is not near zero for more than 80% of rows",
                    "near_zero_share": near_zero_share,
                }
            )

    if "daily_variability_cv" in output.columns:
        variability_seasonality_corr = max_abs_correlation(
            pearson,
            spearman,
            "daily_variability_cv",
            "log_summer_winter_ratio",
        )
        if is_strongly_correlated(pearson, spearman, "daily_variability_cv", "log_summer_winter_ratio"):
            rationale["decisions"].append(
                {
                    "decision": "dropped_daily_variability_cv",
                    "reason": "daily_variability_cv is strongly correlated with log_summer_winter_ratio; keeping interpretable seasonality",
                    "max_abs_correlation": variability_seasonality_corr,
                }
            )
        else:
            selected.append("daily_variability_cv")
            rationale["decisions"].append(
                {
                    "decision": "kept_daily_variability_cv",
                    "reason": "daily_variability_cv is not strongly correlated with log_summer_winter_ratio",
                    "max_abs_correlation": variability_seasonality_corr,
                }
            )

    selected = list(dict.fromkeys(selected))
    selected_table = output[["location_group_id", *selected]].dropna(subset=selected).copy()
    rationale["selected_features"] = selected
    rationale["n_rows_after_core_filter"] = int(len(output))
    rationale["n_rows_in_selected_table"] = int(len(selected_table))
    rationale["dropped_rows_due_to_selected_feature_missingness"] = int(len(output) - len(selected_table))
    return selected_table, selected, rationale


def run_correlation_analysis(
    input_path: Path = DEFAULT_INPUT,
    pearson_output: Path = DEFAULT_PEARSON_OUTPUT,
    spearman_output: Path = DEFAULT_SPEARMAN_OUTPUT,
    flags_output: Path = DEFAULT_FLAGS_OUTPUT,
    selected_json_output: Path = DEFAULT_SELECTED_JSON_OUTPUT,
    selected_table_output: Path = DEFAULT_SELECTED_TABLE_OUTPUT,
) -> dict[str, object]:
    """Run correlation analysis and write all outputs."""
    features = load_features(input_path)
    clean = filter_complete_core(features)
    if clean.empty:
        raise ValueError("No rows remain after removing missing core candidate features.")

    candidate_columns = [column for column in ALL_CANDIDATE_FEATURES if column in clean.columns]
    pearson, spearman = calculate_correlations(clean, candidate_columns)
    flags = build_correlation_flags(pearson, spearman)
    selected_table, selected_features, rationale = recommend_features(clean, pearson, spearman)

    for path in [pearson_output, spearman_output, flags_output, selected_json_output, selected_table_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    pearson.to_csv(pearson_output)
    spearman.to_csv(spearman_output)
    flags.to_csv(flags_output, index=False)
    selected_table.to_csv(selected_table_output, index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.relative_to(PROJECT_ROOT)),
        "outputs": {
            "pearson": str(pearson_output.relative_to(PROJECT_ROOT)),
            "spearman": str(spearman_output.relative_to(PROJECT_ROOT)),
            "flags": str(flags_output.relative_to(PROJECT_ROOT)),
            "selected_table": str(selected_table_output.relative_to(PROJECT_ROOT)),
        },
        "core_candidate_features": CORE_CANDIDATE_FEATURES,
        "optional_features_considered": [column for column in OPTIONAL_FEATURES if column in clean.columns],
        "n_rows_input": int(len(features)),
        "n_rows_after_core_filter": int(len(clean)),
        "n_rows_removed_missing_core": int(len(features) - len(clean)),
        "strong_correlation_threshold": CORRELATION_THRESHOLD,
        "n_flagged_pairs": int(len(flags)),
        **rationale,
    }
    selected_json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze location-group feature correlations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_correlation_analysis(input_path=args.input)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Location-group correlation analysis complete")
    print(f"- rows input: {report['n_rows_input']}")
    print(f"- rows after core feature filter: {report['n_rows_after_core_filter']}")
    print(f"- flagged strong correlation pairs: {report['n_flagged_pairs']}")
    print(f"- selected features: {report['selected_features']}")
    print(f"- outputs: {report['outputs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
