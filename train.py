"""
train.py

Standalone training script for the Employee Attrition Decision Tree model.
Run this once (or whenever you want to retrain) to produce the model
artifacts that app.py loads at startup.

Usage:
    python train.py
    python train.py --max-depth 6 --data data/employee_attrition.csv

Outputs (written to models/):
    model.pkl           - trained DecisionTreeClassifier
    encoders.pkl         - dict of {column_name: fitted LabelEncoder}
    feature_names.pkl    - ordered list of feature columns the model expects
    metrics.json          - accuracy / precision / recall / f1 / roc_auc on the test split
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

DROP_COLS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


def train(data_path: str, output_dir: str, max_depth: int, test_size: float, random_state: int):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    work = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

    encoders = {}
    for col in work.select_dtypes(include="object").columns:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col])
        encoders[col] = le

    X = work.drop("Attrition", axis=1)
    y = work["Attrition"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training DecisionTreeClassifier (max_depth={max_depth}) on {len(X_train)} rows ...")
    model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "max_depth": max_depth,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(output_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(output_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts to:", output_dir.resolve())
    print("\nTest set performance:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:10s}: {v:.4f}")
        else:
            print(f"  {k:10s}: {v}")

    return model, encoders, feature_names, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Employee Attrition Decision Tree model.")
    parser.add_argument("--data", default="data/employee_attrition.csv", help="Path to the CSV dataset")
    parser.add_argument("--output-dir", default="models", help="Directory to save model artifacts")
    parser.add_argument("--max-depth", type=int, default=5, help="Decision tree max depth")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state,
    )
