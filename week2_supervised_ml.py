"""
Week 2: Supervised Machine Learning Models
==========================================

Regression:
    Scikit-learn Diabetes dataset

Classification:
    Scikit-learn Breast Cancer dataset

Models:
    Regression    - Linear Regression, Decision Tree, Random Forest, KNN
    Classification - Logistic Regression, Decision Tree, Random Forest, KNN

Outputs:
    - Prints model evaluation tables to the terminal
    - Saves regression_results.csv
    - Saves classification_results.csv

Install dependencies:
    python -m pip install numpy pandas scikit-learn

Run:
    python week2_supervised_ml.py
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path(".")  # Results are saved in the current working directory.


# ---------------------------------------------------------------------
# Regression: Diabetes dataset
# ---------------------------------------------------------------------

def evaluate_regression_models():
    """Train regression models and return their evaluation results."""
    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=7),
        ),
    }

    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mean_squared_error(y_test, predictions),
                "R2": r2_score(y_test, predictions),
            }
        )

    results = pd.DataFrame(rows).set_index("Model")
    return results.sort_values("R2", ascending=False).round(4)


# ---------------------------------------------------------------------
# Classification: Breast Cancer dataset
# ---------------------------------------------------------------------

def evaluate_classification_models():
    """Train classification models and return their evaluation results."""
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),
    }

    rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, predictions, zero_division=0
                ),
            }
        )

    results = pd.DataFrame(rows).set_index("Model")
    return results.sort_values("F1-score", ascending=False).round(4)


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():
    """Run both experiments, print results, and save CSV reports."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    regression_results = evaluate_regression_models()
    classification_results = evaluate_classification_models()

    print("\n" + "=" * 72)
    print("REGRESSION RESULTS — DIABETES DATASET")
    print("=" * 72)
    print(regression_results.to_string())

    print("\n" + "=" * 72)
    print("CLASSIFICATION RESULTS — BREAST CANCER DATASET")
    print("=" * 72)
    print(classification_results.to_string())

    regression_path = OUTPUT_DIR / "regression_results.csv"
    classification_path = OUTPUT_DIR / "classification_results.csv"

    regression_results.to_csv(regression_path)
    classification_results.to_csv(classification_path)

    print("\n" + "=" * 72)
    print("RESULT FILES SAVED")
    print("=" * 72)
    print(f"- {regression_path.resolve()}")
    print(f"- {classification_path.resolve()}")


if __name__ == "__main__":
    main()
