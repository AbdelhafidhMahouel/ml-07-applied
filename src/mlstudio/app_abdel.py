"""app_abdel.py - custom version of app_case.py.

An example of a supervised regression case, modified by Abdelhafidh Mahouel.
Based on the original working example, app_case.py, by Denise Case.

Author: Abdelhafidh Mahouel
Date: 2026-08

Process:
    - Load a CSV dataset.
    - Engineer a new derived feature (study_efficiency).
    - Train and compare TWO supervised regression models (Linear Regression
      and Ridge Regression) instead of just one.
    - Evaluate and log model performance for both models.
    - Predict one new custom case (different values than the original example).
    - Create three charts instead of two, including two new chart types
      that were not in the original example (a correlation heatmap and a
      regression line plot with a trend line).

Data Source:
- data/raw/hours_scores_case.csv

Terminal command to run this file from the root project folder:

uv run python -m mlstudio.app_abdel

Summary of modifications from app_case.py:
    1. Added a new derived feature, study_efficiency (hours_studied / sleep_hours),
       and included it in the model's feature set.
    2. Trained and compared two models (LinearRegression and Ridge) instead of one,
       and logged both sets of metrics so they can be compared directly.
    3. Changed the new prediction case to different, custom input values.
    4. Replaced the plain scatter plot with a seaborn regplot (adds a trend line).
    5. Replaced the coefficient bar chart color scheme and added a third,
       brand-new chart: a correlation heatmap of all features and the target.
    6. Added extra LOG.info() observability statements, including the
       correlation between study_efficiency and score.
"""

# === Section 1a. DECLARE IMPORTS (BRING IN FREE CODE) ===

import logging
from typing import Final

from datafun_toolkit.logger import get_logger, log_header
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# === Section 1b. CONFIGURE LOGGER ONCE PER MODULE ===

LOG: logging.Logger = get_logger("ML", level="DEBUG")
log_header(LOG, "ML")

# === Section 1c. Global Constants and Configuration ===

DATASET_NAME: Final[str] = "hours_scores_case"

TARGET_COL: Final[str] = "score"

# MODIFICATION 1: Added a new derived feature, study_efficiency, to the
# feature list. This feature is engineered in make_clean_view() below.
FEATURE_COLS: Final[list[str]] = [
    "hours_studied",
    "practice_quizzes",
    "attendance_pct",
    "sleep_hours",
    "prior_score",
    "study_efficiency",
]

TEST_SIZE: Final[float] = 0.30
RANDOM_STATE: Final[int] = 42

# MODIFICATION 2: Added a Ridge regularization strength constant so we can
# train and compare a second model alongside the original LinearRegression.
RIDGE_ALPHA: Final[float] = 1.0

# === Section 1d. Pandas Configuration for Display ===

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)


# === Section 2. Load the Data ===


def load_data() -> pd.DataFrame:
    """Load the case dataset from the data/raw folder."""
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


# === Section 4. Check Data Quality ===


def check_quality(df: pd.DataFrame) -> None:
    """Check missing values and duplicate rows."""
    LOG.info("Missing values by column")
    LOG.debug(f"\n{df.isna().sum()}")

    duplicate_count: int = df.duplicated().sum()
    LOG.info(f"Duplicate row count: {duplicate_count}")


# === Section 5. Create a Clean View ===


def make_clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a cleaned view for modeling, with one new engineered feature.

    MODIFICATION 1 (continued): study_efficiency is computed here as
    hours_studied divided by sleep_hours. The idea is to capture how much a
    student studies relative to how much they sleep, rather than just
    looking at hours_studied alone.
    """
    LOG.info("Creating clean modeling view")

    df_working: pd.DataFrame = df.copy()

    # New engineered feature: study efficiency.
    df_working["study_efficiency"] = (
        df_working["hours_studied"] / df_working["sleep_hours"]
    )
    LOG.info("Engineered new feature: study_efficiency = hours_studied / sleep_hours")

    selected_cols: list[str] = FEATURE_COLS + [TARGET_COL]

    df_selected: pd.DataFrame = df_working[selected_cols]  # type: ignore[assignment]
    df_no_missing: pd.DataFrame = df_selected.dropna()
    df_clean: pd.DataFrame = df_no_missing.copy()

    LOG.info(f"Clean view: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    # MODIFICATION 6: extra observability - log the correlation between the
    # new engineered feature and the target.
    corr_value: float = df_clean["study_efficiency"].corr(df_clean[TARGET_COL])
    LOG.info(f"Correlation of study_efficiency with score: {corr_value:.3f}")

    return df_clean


# === Section 6. Train and Compare Supervised Models ===


def train_models(
    df_clean: pd.DataFrame,
) -> tuple[LinearRegression, Ridge]:
    """Train and compare TWO supervised regression models.

    MODIFICATION 2: Instead of training only a LinearRegression model, this
    version also trains a Ridge regression model on the same split, so we
    can directly compare their error and R-squared side by side.
    """
    LOG.info("Training LinearRegression model")

    x = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)
    linear_pred = linear_model.predict(x_test)

    linear_mae: float = mean_absolute_error(y_test, linear_pred)
    linear_r2: float = r2_score(y_test, linear_pred)

    LOG.info(f"LinearRegression Mean absolute error: {linear_mae:.2f}")
    LOG.info(f"LinearRegression R-squared: {linear_r2:.2f}")

    LOG.info(f"Training Ridge model (alpha={RIDGE_ALPHA})")

    ridge_model = Ridge(alpha=RIDGE_ALPHA)
    ridge_model.fit(x_train, y_train)
    ridge_pred = ridge_model.predict(x_test)

    ridge_mae: float = mean_absolute_error(y_test, ridge_pred)
    ridge_r2: float = r2_score(y_test, ridge_pred)

    LOG.info(f"Ridge Mean absolute error: {ridge_mae:.2f}")
    LOG.info(f"Ridge R-squared: {ridge_r2:.2f}")

    if linear_mae <= ridge_mae:
        LOG.info("Comparison: LinearRegression had the lower (better) MAE.")
    else:
        LOG.info("Comparison: Ridge had the lower (better) MAE.")

    return linear_model, ridge_model


# === Section 7. Predict One New Case ===


def predict_example(model: LinearRegression) -> None:
    """Use the trained model to predict one new, custom student score.

    MODIFICATION 3: Uses different input values than the original example
    case, representing a lower-effort student profile (fewer study hours,
    fewer practice quizzes, lower attendance) to see how the model responds
    at the opposite end of the range from the original example.
    """
    LOG.info("Predicting one new custom case")

    new_case = pd.DataFrame(
        [
            {
                "hours_studied": 2.5,
                "practice_quizzes": 1,
                "attendance_pct": 70,
                "sleep_hours": 5.5,
                "prior_score": 60,
            }
        ]
    )
    new_case["study_efficiency"] = new_case["hours_studied"] / new_case["sleep_hours"]

    predicted_score: float = model.predict(new_case)[0]

    LOG.info(f"New case:\n{new_case}")
    LOG.info(f"Predicted score: {predicted_score:.1f}")


# === Section 8. Create Visualizations ===


def make_plots(
    df_clean: pd.DataFrame, linear_model: LinearRegression, ridge_model: Ridge
) -> None:
    """Create charts for the supervised regression case.

    MODIFICATION 4: The hours-studied-vs-score scatter plot is replaced with
    a seaborn regplot, which adds a fitted trend line on top of the points.

    MODIFICATION 5: The coefficient bar chart now uses a different color
    palette (viridis) instead of the default seaborn color, and a brand-new
    third chart, a correlation heatmap, has been added that did not exist
    in the original example.
    """
    LOG.info("Creating chart: hours studied vs score (with trend line)")

    fig, ax = plt.subplots(figsize=(9, 5))

    reg_plt: Axes = sns.regplot(
        data=df_clean,
        x="hours_studied",
        y=TARGET_COL,
        ax=ax,
        scatter_kws={"color": "darkorange"},
        line_kws={"color": "navy"},
    )

    reg_plt.set_title(
        "Hours Studied vs Score with Trend Line (CLOSE chart to continue)"
    )
    reg_plt.set_xlabel("Hours Studied")
    reg_plt.set_ylabel("Score")

    LOG.info("Creating chart: model coefficients (LinearRegression vs Ridge)")

    fig, ax = plt.subplots(figsize=(9, 5))

    LOG.info(f"Got a figure {fig} and axes {ax} from plt.subplots().")

    coefficient_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS * 2,
            "coefficient": list(linear_model.coef_) + list(ridge_model.coef_),
            "model": ["LinearRegression"] * len(FEATURE_COLS)
            + ["Ridge"] * len(FEATURE_COLS),
        }
    )

    bar_plt: Axes = sns.barplot(
        data=coefficient_df,
        x="coefficient",
        y="feature",
        hue="model",
        ax=ax,
        palette="viridis",
    )

    bar_plt.set_title(
        "Model Coefficients: LinearRegression vs Ridge (CLOSE chart to continue)"
    )
    bar_plt.set_xlabel("Coefficient")
    bar_plt.set_ylabel("Feature")

    LOG.info("Creating chart: correlation heatmap (NEW chart, not in original example)")

    fig, ax = plt.subplots(figsize=(8, 6))

    corr_matrix = df_clean.corr(numeric_only=True)

    heat_plt: Axes = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
    )

    heat_plt.set_title(
        "Correlation Heatmap of Features and Score (CLOSE chart to continue)"
    )


# === Section 9. Summary and Next Steps ===


def summarize(df: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Log a brief summary."""
    LOG.info("========================")
    LOG.info("SUMMARY")
    LOG.info("========================")
    LOG.info(f"Dataset: {DATASET_NAME}")
    LOG.info(f"Original rows: {df.shape[0]}")
    LOG.info(f"Clean rows: {df_clean.shape[0]}")
    LOG.info(f"Features: {FEATURE_COLS}")
    LOG.info(f"Target: {TARGET_COL}")
    LOG.info("Modifications: added study_efficiency feature, compared")
    LOG.info("LinearRegression vs Ridge, new custom prediction case,")
    LOG.info("regplot with trend line, and a new correlation heatmap chart.")


# === DEFINE THE MAIN FUNCTION THAT CALLS OTHER FUNCTIONS ===


def main() -> None:
    """Main function to run the modified supervised ML workflow."""
    log_header(LOG, "ML")

    LOG.info("========================")
    LOG.info("START main() - abdel custom version")
    LOG.info("========================")

    LOG.info("Load dataset..............")
    df = load_data()

    LOG.info("Inspect dataset...........")
    inspect_basic(df)

    LOG.info("Check data quality........")
    check_quality(df)

    LOG.info("Create clean view (with new feature)..........")
    df_clean = make_clean_view(df)

    LOG.info("Train and compare two supervised models....")
    linear_model, ridge_model = train_models(df_clean)

    LOG.info("Predict one new custom case..........")
    predict_example(linear_model)

    LOG.info("Create charts (regplot, grouped bar, new heatmap)...")
    make_plots(df_clean, linear_model, ridge_model)

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
