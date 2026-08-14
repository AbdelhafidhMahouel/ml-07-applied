"""app_manufacturing_abdel.py - Phase 5 custom project.

Applies the skills from the Module 7 example (load data, engineer features,
train and evaluate supervised models, predict new cases, and visualize
results) to a new domain: injection molding process engineering in
manufacturing.

Author: Abdelhafidh Mahouel
Date: 2026-08

Problem framing:
    A plastics manufacturer records process parameters for each production
    batch on an injection molding line (melt temperature, injection
    pressure, cooling time, material moisture, operator experience, etc.).
    Quality engineers want two things from a data-driven system:

    1. CLASSIFICATION: predict whether a batch is likely to be defective
       (defect = 1) BEFORE it happens, based on the process settings, so
       operators can adjust settings proactively rather than discovering
       defects after the fact.
    2. REGRESSION: predict the expected part weight (grams) for a batch,
       since part weight is a continuous quality metric tied directly to
       dimensional accuracy and material cost.

This is a different domain (manufacturing process engineering) and a
different problem shape (one classification target AND one regression
target, using a larger, richer 19-column / 3000-row dataset) than the
Module 7 example, which only investigated an already-deployed external
regression-style classification API. Here, we train, evaluate, and
interrogate our OWN models locally, end to end.

Data Source:
- data/raw/manufacturing_quality_abdel.csv (synthetic but realistically
  engineered injection molding process data; see data/raw/README.md)

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_manufacturing_abdel

Process:
    - Load the manufacturing process dataset.
    - Inspect and validate data quality.
    - Engineer two new features (temperature deviation from setpoint,
      cooling adequacy ratio).
    - Train and compare TWO classifiers to predict defects
      (RandomForestClassifier and GradientBoostingClassifier).
    - Train a RandomForestRegressor to predict part weight (grams).
    - Evaluate both tasks with appropriate professional metrics
      (accuracy, precision, recall, F1, ROC-AUC for classification;
      MAE and R-squared for regression).
    - Predict outcomes for two new, realistic example batches
      (one well-controlled, one poorly-controlled process run).
    - Probe the trained defect classifier the same way the Module 7
      notebook probed the deployed penguin API: sweep one feature at a
      time, build a two-feature decision grid, and test edge cases.
    - Create six professional diagnostic charts.
"""

# === Section 1a. DECLARE IMPORTS ===

import logging
from typing import Any, Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("MFG", level="DEBUG")
log_header(LOG, "MFG")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "manufacturing_quality_abdel"

CLASSIFICATION_TARGET: Final[str] = "defect"
REGRESSION_TARGET: Final[str] = "part_weight_g"

NUMERIC_FEATURE_COLS: Final[list[str]] = [
    "operator_experience_years",
    "melt_temperature_c",
    "mold_temperature_c",
    "injection_pressure_bar",
    "injection_speed_mm_s",
    "screw_speed_rpm",
    "back_pressure_bar",
    "hold_pressure_bar",
    "hold_time_s",
    "cooling_time_s",
    "cycle_time_s",
    "material_moisture_pct",
    "ambient_humidity_pct",
]

# Engineered features (see engineer_features()).
ENGINEERED_FEATURE_COLS: Final[list[str]] = [
    "melt_temp_deviation",
    "cooling_adequacy_ratio",
]

MODEL_FEATURE_COLS: Final[list[str]] = NUMERIC_FEATURE_COLS + ENGINEERED_FEATURE_COLS

TEST_SIZE: Final[float] = 0.25
RANDOM_STATE: Final[int] = 7

# Process engineering reference setpoints (used for feature engineering and
# for the "well-controlled batch" example prediction).
IDEAL_MELT_TEMPERATURE_C: Final[float] = 235.0
MIN_ADEQUATE_COOLING_S: Final[float] = 14.0

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 130)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the manufacturing process dataset from the data/raw folder."""
    LOG.info(f"Loading dataset: {DATASET_NAME}")

    df: pd.DataFrame = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")

    LOG.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.debug(f"\n{df.head()}")

    return df


# === Section 3. Inspect Data Shape and Structure ===


def inspect_basic(df: pd.DataFrame) -> None:
    """Inspect basic dataset structure."""
    LOG.info("Column names")
    LOG.debug(f"{list(df.columns)}")

    LOG.info("DataFrame info")
    df.info()

    LOG.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    LOG.info(f"Defect rate: {df[CLASSIFICATION_TARGET].mean():.1%}")
    LOG.info(f"Machines represented: {sorted(df['machine_id'].unique())}")
    LOG.info(f"Shifts represented: {sorted(df['shift'].unique())}")


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    LOG.debug(f"\n{df.isna().sum()}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")

    duplicate_batch_ids: int = df["batch_id"].duplicated().sum()
    LOG.info(f"Duplicate batch_id count: {duplicate_batch_ids}")


# === Section 5. Feature Engineering ===


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add two new engineered features that reflect process engineering judgment.

    WHY: Raw sensor readings alone do not tell an engineer whether a
    process is "in control." Two derived signals are more directly tied to
    defect risk than the raw readings on their own:

    - melt_temp_deviation: absolute distance from the ideal melt
      temperature setpoint. Both too hot and too cold cause different
      defects, so the RAW temperature is less informative than its
      DEVIATION from target.
    - cooling_adequacy_ratio: how much cooling time was allowed relative
      to the minimum adequate cooling time. A ratio below 1.0 signals an
      under-cooled (warpage-prone) part.
    """
    df_out: pd.DataFrame = df.copy()

    df_out["melt_temp_deviation"] = (
        df_out["melt_temperature_c"] - IDEAL_MELT_TEMPERATURE_C
    ).abs()

    df_out["cooling_adequacy_ratio"] = df_out["cooling_time_s"] / MIN_ADEQUATE_COOLING_S

    LOG.info("Engineered feature: melt_temp_deviation = |melt_temperature_c - 235|")
    LOG.info(
        "Engineered feature: cooling_adequacy_ratio = cooling_time_s / 14.0 seconds"
    )

    return df_out


# === Section 6. Create a Clean Modeling View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view for modeling."""
    LOG.info("Creating clean modeling view")

    selected_cols: list[str] = MODEL_FEATURE_COLS + [
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
    ]

    df_selected: pd.DataFrame = df[selected_cols]  # type: ignore[assignment]
    df_no_missing: pd.DataFrame = df_selected.dropna()
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    return df_clean


# === Section 7. Train and Evaluate Classification Models ===


def train_classifiers(
    df_clean: pd.DataFrame,
) -> tuple[RandomForestClassifier, GradientBoostingClassifier, pd.DataFrame, pd.Series]:
    """Train and compare two classifiers that predict the defect target.

    Returns the two fitted models plus the held-out test split, so the
    caller can build additional charts (confusion matrix, ROC curve)
    without retraining.
    """
    LOG.info("Training classifiers to predict: defect (1 = likely defective)")

    x = df_clean[MODEL_FEATURE_COLS]
    y = df_clean[CLASSIFICATION_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE
    )
    rf_model.fit(x_train, y_train)
    rf_pred = rf_model.predict(x_test)
    rf_proba = rf_model.predict_proba(x_test)[:, 1]

    LOG.info("RandomForestClassifier results:")
    LOG.info(f"  Accuracy:  {accuracy_score(y_test, rf_pred):.3f}")
    LOG.info(f"  Precision: {precision_score(y_test, rf_pred):.3f}")
    LOG.info(f"  Recall:    {recall_score(y_test, rf_pred):.3f}")
    LOG.info(f"  F1 score:  {f1_score(y_test, rf_pred):.3f}")
    LOG.info(f"  ROC-AUC:   {roc_auc_score(y_test, rf_proba):.3f}")

    gb_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=RANDOM_STATE
    )
    gb_model.fit(x_train, y_train)
    gb_pred = gb_model.predict(x_test)
    gb_proba = gb_model.predict_proba(x_test)[:, 1]

    LOG.info("GradientBoostingClassifier results:")
    LOG.info(f"  Accuracy:  {accuracy_score(y_test, gb_pred):.3f}")
    LOG.info(f"  Precision: {precision_score(y_test, gb_pred):.3f}")
    LOG.info(f"  Recall:    {recall_score(y_test, gb_pred):.3f}")
    LOG.info(f"  F1 score:  {f1_score(y_test, gb_pred):.3f}")
    LOG.info(f"  ROC-AUC:   {roc_auc_score(y_test, gb_proba):.3f}")

    if roc_auc_score(y_test, rf_proba) >= roc_auc_score(y_test, gb_proba):
        LOG.info("Comparison: RandomForestClassifier had the higher (better) ROC-AUC.")
    else:
        LOG.info(
            "Comparison: GradientBoostingClassifier had the higher (better) ROC-AUC."
        )

    return rf_model, gb_model, x_test, y_test


# === Section 8. Train and Evaluate the Regression Model ===


def train_regressor(
    df_clean: pd.DataFrame,
) -> RandomForestRegressor:
    """Train a regressor that predicts continuous part weight in grams."""
    LOG.info("Training RandomForestRegressor to predict: part_weight_g")

    x = df_clean[MODEL_FEATURE_COLS]
    y = df_clean[REGRESSION_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    reg_model = RandomForestRegressor(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE
    )
    reg_model.fit(x_train, y_train)
    y_pred = reg_model.predict(x_test)

    mae: float = mean_absolute_error(y_test, y_pred)
    r2: float = r2_score(y_test, y_pred)

    LOG.info(f"RandomForestRegressor Mean absolute error: {mae:.3f} grams")
    LOG.info(f"RandomForestRegressor R-squared: {r2:.3f}")

    return reg_model


# === Section 9. Predict Two New Example Batches ===


def build_batch_row(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame with engineered features from raw inputs.

    WHY: Any new batch used for prediction must go through the SAME feature
    engineering as the training data, or the model will see inconsistent
    inputs. Centralizing this logic in one function avoids that mistake.
    """
    row = pd.DataFrame([payload])
    row["melt_temp_deviation"] = (
        row["melt_temperature_c"] - IDEAL_MELT_TEMPERATURE_C
    ).abs()
    row["cooling_adequacy_ratio"] = row["cooling_time_s"] / MIN_ADEQUATE_COOLING_S
    return row[MODEL_FEATURE_COLS]


def predict_example_batches(
    clf_model: RandomForestClassifier, reg_model: RandomForestRegressor
) -> None:
    """Predict outcomes for a well-controlled batch and a poorly-controlled batch."""
    LOG.info("Predicting two new example batches")

    well_controlled: dict[str, Any] = {
        "operator_experience_years": 12.0,
        "melt_temperature_c": 235.0,
        "mold_temperature_c": 55.0,
        "injection_pressure_bar": 120.0,
        "injection_speed_mm_s": 55.0,
        "screw_speed_rpm": 115.0,
        "back_pressure_bar": 14.0,
        "hold_pressure_bar": 70.0,
        "hold_time_s": 5.5,
        "cooling_time_s": 18.0,
        "cycle_time_s": 30.0,
        "material_moisture_pct": 0.04,
        "ambient_humidity_pct": 48.0,
    }

    poorly_controlled: dict[str, Any] = {
        "operator_experience_years": 1.5,
        "melt_temperature_c": 254.0,
        "mold_temperature_c": 70.0,
        "injection_pressure_bar": 140.0,
        "injection_speed_mm_s": 98.0,
        "screw_speed_rpm": 180.0,
        "back_pressure_bar": 22.0,
        "hold_pressure_bar": 80.0,
        "hold_time_s": 4.0,
        "cooling_time_s": 8.0,
        "cycle_time_s": 20.0,
        "material_moisture_pct": 0.18,
        "ambient_humidity_pct": 65.0,
    }

    for label, payload in [
        ("Well-controlled batch", well_controlled),
        ("Poorly-controlled batch", poorly_controlled),
    ]:
        row = build_batch_row(payload)
        defect_pred = int(clf_model.predict(row)[0])
        defect_proba = float(clf_model.predict_proba(row)[0][1])
        weight_pred = float(reg_model.predict(row)[0])

        LOG.info(f"{label}:")
        LOG.info(
            f"  Predicted defect: {defect_pred} (probability of defect: {defect_proba:.2f})"
        )
        LOG.info(
            f"  Predicted part weight: {weight_pred:.2f} g (nominal target: 50.0 g)"
        )


# === Section 10. Probe the Classifier Like a Deployed Model ===
# WHY: This mirrors the Module 7 example's approach to investigating a
# deployed API from the outside (sweep a feature, build a 2D grid, test
# edge cases), applied here to a model we trained ourselves.


def sweep_feature(
    clf_model: RandomForestClassifier,
    base: dict[str, Any],
    feature: str,
    values: list[float],
) -> pd.DataFrame:
    """Vary one process parameter across a range and collect defect probabilities."""
    rows = []
    for v in values:
        payload = {**base, feature: v}
        row = build_batch_row(payload)
        proba = float(clf_model.predict_proba(row)[0][1])
        rows.append({feature: v, "defect_probability": proba})
    return pd.DataFrame(rows)


def test_edge_cases(clf_model: RandomForestClassifier, base: dict[str, Any]) -> None:
    """Test unusual or extreme process settings against the trained classifier."""
    edge_cases: list[tuple[str, dict[str, Any]]] = [
        ("extremely hot melt", {**base, "melt_temperature_c": 265.0}),
        ("extremely cold melt", {**base, "melt_temperature_c": 200.0}),
        ("near-zero cooling time", {**base, "cooling_time_s": 4.0}),
        ("very high material moisture", {**base, "material_moisture_pct": 0.35}),
        ("brand-new operator (0 years)", {**base, "operator_experience_years": 0.0}),
        (
            "everything at worst case",
            {
                **base,
                "melt_temperature_c": 265.0,
                "cooling_time_s": 4.0,
                "material_moisture_pct": 0.35,
                "injection_speed_mm_s": 110.0,
                "operator_experience_years": 0.0,
            },
        ),
    ]

    LOG.info("Edge case results (trained classifier, not an external API):")
    for label, payload in edge_cases:
        row = build_batch_row(payload)
        proba = float(clf_model.predict_proba(row)[0][1])
        pred = int(clf_model.predict(row)[0])
        LOG.info(f"  {label:<32} -> defect={pred}  probability={proba:.2f}")


# === Section 11. Create Visualizations ===


def make_plots(
    df_clean: pd.DataFrame,
    rf_model: RandomForestClassifier,
    gb_model: GradientBoostingClassifier,
    reg_model: RandomForestRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Create six professional diagnostic charts for the manufacturing project."""

    # --- Chart 1: Correlation heatmap of numeric features and both targets ---
    LOG.info("Creating chart 1: correlation heatmap")
    plt.figure(figsize=(11, 9))
    corr_matrix = df_clean.corr(numeric_only=True)
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    plt.title("Correlation Heatmap: Process Parameters, Defect, and Part Weight")
    plt.tight_layout()

    # --- Chart 2: Feature importance (RandomForestClassifier) ---
    LOG.info("Creating chart 2: classifier feature importance")
    importance_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLS,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(9, 6))
    bar_plt: Axes = sns.barplot(
        data=importance_df,
        x="importance",
        y="feature",
        hue="feature",
        palette="mako",
        legend=False,
    )
    bar_plt.set_title("RandomForestClassifier Feature Importance (Defect Prediction)")
    bar_plt.set_xlabel("Importance")
    bar_plt.set_ylabel("Feature")
    plt.tight_layout()

    # --- Chart 3: Confusion matrix (RandomForestClassifier) ---
    LOG.info("Creating chart 3: confusion matrix")
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(
        rf_model,
        x_test,
        y_test,
        display_labels=["No Defect", "Defect"],
        cmap="Blues",
        ax=ax3,
    )
    ax3.set_title("Confusion Matrix: RandomForestClassifier (Test Set)")
    plt.tight_layout()

    # --- Chart 4: ROC curve comparison for both classifiers ---
    LOG.info("Creating chart 4: ROC curve comparison")
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(rf_model, x_test, y_test, ax=ax, name="RandomForest")
    RocCurveDisplay.from_estimator(
        gb_model, x_test, y_test, ax=ax, name="GradientBoosting"
    )
    ax.set_title("ROC Curve Comparison: Defect Classifiers")
    plt.tight_layout()

    # --- Chart 5: Defect rate by machine and shift (grouped bar) ---
    LOG.info("Creating chart 5: defect rate by machine and shift")
    # Re-attach machine_id/shift for this summary view only.
    df_full = pd.read_csv(f"data/raw/{DATASET_NAME}.csv")
    summary = df_full.groupby(["machine_id", "shift"])["defect"].mean().reset_index()
    plt.figure(figsize=(9, 6))
    grp_plt: Axes = sns.barplot(
        data=summary, x="machine_id", y="defect", hue="shift", palette="rocket"
    )
    grp_plt.set_title("Defect Rate by Machine and Shift")
    grp_plt.set_xlabel("Machine")
    grp_plt.set_ylabel("Defect Rate")
    plt.tight_layout()

    # --- Chart 6: Melt temperature deviation vs defect probability (sweep) ---
    LOG.info("Creating chart 6: melt temperature sweep vs defect probability")
    base_case: dict[str, Any] = {
        "operator_experience_years": 7.0,
        "melt_temperature_c": 235.0,
        "mold_temperature_c": 55.0,
        "injection_pressure_bar": 120.0,
        "injection_speed_mm_s": 60.0,
        "screw_speed_rpm": 120.0,
        "back_pressure_bar": 14.0,
        "hold_pressure_bar": 70.0,
        "hold_time_s": 5.5,
        "cooling_time_s": 16.0,
        "cycle_time_s": 27.0,
        "material_moisture_pct": 0.06,
        "ambient_humidity_pct": 50.0,
    }
    melt_values = list(np.linspace(205, 265, 25))
    df_sweep = sweep_feature(rf_model, base_case, "melt_temperature_c", melt_values)

    LOG.info("melt_temperature_c sweep (defect probability):")
    LOG.info(df_sweep.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.plot(
        df_sweep["melt_temperature_c"],
        df_sweep["defect_probability"],
        marker="o",
        color="firebrick",
    )
    plt.axhline(0.5, color="gray", linestyle="--", label="0.5 decision threshold")
    plt.axvline(235, color="steelblue", linestyle=":", label="Ideal setpoint (235C)")
    plt.xlabel("melt_temperature_c")
    plt.ylabel("Predicted defect probability")
    plt.title("Feature Sensitivity: Defect Probability vs Melt Temperature")
    plt.legend()
    plt.tight_layout()


# === Section 12. Summary and Next Steps ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Log a brief professional summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows used for modeling: {df_clean.shape[0]}")
    LOG.info(f"Model features: {MODEL_FEATURE_COLS}")
    LOG.info(f"Classification target: {CLASSIFICATION_TARGET}")
    LOG.info(f"Regression target: {REGRESSION_TARGET}")
    LOG.info("Applied the Module 7 example's approach (train, evaluate, predict new")
    LOG.info("cases, then probe the model with feature sweeps and edge cases) to a")
    LOG.info("new manufacturing process engineering problem.")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the manufacturing process engineering workflow."""
    log_header(LOG, "MFG")

    LOG.info("========================")
    LOG.info("START main() - Phase 5 custom manufacturing project")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Engineer features..........")
    df_engineered = engineer_features(df)

    LOG.info("Create clean modeling view..........")
    df_clean = make_clean_view(df_engineered)

    LOG.info("Train and compare classifiers (defect prediction)....")
    rf_model, gb_model, x_test, y_test = train_classifiers(df_clean)

    LOG.info("Train regressor (part weight prediction)....")
    reg_model = train_regressor(df_clean)

    LOG.info("Predict two new example batches..........")
    predict_example_batches(rf_model, reg_model)

    LOG.info("Test edge cases against the trained classifier..........")
    base_case: dict[str, Any] = {
        "operator_experience_years": 7.0,
        "melt_temperature_c": 235.0,
        "mold_temperature_c": 55.0,
        "injection_pressure_bar": 120.0,
        "injection_speed_mm_s": 60.0,
        "screw_speed_rpm": 120.0,
        "back_pressure_bar": 14.0,
        "hold_pressure_bar": 70.0,
        "hold_time_s": 5.5,
        "cooling_time_s": 16.0,
        "cycle_time_s": 27.0,
        "material_moisture_pct": 0.06,
        "ambient_humidity_pct": 50.0,
    }
    test_edge_cases(rf_model, base_case)

    LOG.info("Create charts.............")
    make_plots(df_clean, rf_model, gb_model, reg_model, x_test, y_test)

    LOG.info("Summarize workflow........")
    summarize(df, df_clean)

    LOG.info(
        "----- in a script, call plt.show() once at the end to display all charts -----"
    )
    LOG.info(
        "----- in a script, CLOSE the chart windows with the close button to CONTINUE -----"
    )

    plt.show()

    LOG.info("Workflow complete")
    LOG.info("IMPORTANT: This script creates chart windows.")
    LOG.info("Close chart windows and terminate this process with CTRL+c as needed.")
    LOG.info("========================")
    LOG.info("Executed successfully!")
    LOG.info("========================")


# === CONDITIONAL EXECUTION GUARD ===

if __name__ == "__main__":
    main()
