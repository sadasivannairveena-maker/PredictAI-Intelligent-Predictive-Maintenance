import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"

FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear"
]


def load_data():
    print("Downloading UCI AI4I 2020 dataset...")

    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features[FEATURES].copy()
    y = dataset.data.targets["Machine failure"].copy()

    print("Dataset shape:", X.shape)
    print("Failure records:", int(y.sum()))

    return X, y


def build_preprocessor():
    numeric = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear"
    ]

    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         ["Type"])
    ])


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED
        )
    }


def compare_models(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    scoring = {
        "F1": "f1",
        "Recall": "recall",
        "Precision": "precision",
        "ROC_AUC": "roc_auc",
        "PR_AUC": "average_precision"
    }

    results = []
    pipelines = {}

    for name, classifier in build_models().items():
        print(f"\nCross-validating {name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        row = {"Model": name}

        for metric in scoring:
            row[metric] = float(
                scores[f"test_{metric}"].mean()
            )

        results.append(row)
        pipelines[name] = pipeline

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "PR_AUC", ascending=False
    ).reset_index(drop=True)

    return results_df, pipelines


def optimize_threshold(pipeline, X_train, y_train):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train
    )

    candidate = clone(pipeline)
    candidate.fit(X_fit, y_fit)

    probabilities = candidate.predict_proba(X_val)[:, 1]

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.96, 0.05):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_val, predictions, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = round(float(threshold), 2)

    print("Selected threshold:", best_threshold)
    print("Validation F1:", round(best_f1, 4))

    return best_threshold


def evaluate(model, threshold, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist()
    }


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    results_df, pipelines = compare_models(
        X_train, y_train
    )

    print("\nMODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    print("\nSelected model:", best_name)

    best_pipeline = pipelines[best_name]

    threshold = optimize_threshold(
        best_pipeline, X_train, y_train
    )

    final_model = clone(best_pipeline)
    final_model.fit(X_train, y_train)

    metrics = evaluate(
        final_model, threshold, X_test, y_test
    )

    metrics["model"] = best_name
    metrics["threshold"] = threshold
    metrics["dataset"] = "UCI AI4I 2020 synthetic"
    metrics["evaluation_status"] = (
        "Exploratory: existing test split "
        "has been inspected during development"
    )
    metrics["python_version"] = platform.python_version()
    metrics["sklearn_version"] = sklearn.__version__

    joblib.dump(
        final_model, ARTIFACTS / "repro_model.joblib"
    )
    joblib.dump(
        threshold, ARTIFACTS / "repro_threshold.joblib"
    )
    joblib.dump(
        FEATURES, ARTIFACTS / "repro_features.joblib"
    )

    with open(
        ARTIFACTS / "repro_metrics.json", "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    results_df.to_csv(
        REPORTS / "repro_model_comparison.csv",
        index=False
    )

    print("\nFINAL EXPLORATORY EVALUATION")
    print(json.dumps(metrics, indent=4))

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
