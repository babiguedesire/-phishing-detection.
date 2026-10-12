"""Phishing website detection: training, evaluation and robustness test.

Usage:
    python src/train.py                # uses data/phishing.csv
    python src/train.py --data PATH    # custom CSV

Outputs (in results/): metrics.json, metrics_table.csv, confusion_matrix.png,
feature_importance.png, robustness.png, model.joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "Result"  # UCI convention: 1 = legitimate, -1 = phishing


def load_data(path):
    df = pd.read_csv(path)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # avoid train/test leakage
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == -1).astype(int)  # 1 = phishing (positive class)
    return X, y, n_raw, len(df)


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def robustness_test(model, X_test, y_test, top_features, out_png):
    """Evasion simulation: an attacker forges the k most important features of
    phishing pages so they look legitimate (+1). Measures recall on phishing."""
    phish = X_test[y_test == 1]
    ks = list(range(0, len(top_features) + 1))
    recalls = []
    for k in ks:
        Xp = phish.copy()
        for f in top_features[:k]:
            Xp[f] = 1
        recalls.append(float((model.predict(Xp) == 1).mean()))
    plt.figure(figsize=(6, 4))
    plt.plot(ks, recalls, marker="o")
    plt.xlabel("Number of top features forged by attacker (k)")
    plt.ylabel("Phishing recall")
    plt.title("Robustness to feature forgery (Random Forest)")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return dict(zip(ks, recalls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/phishing.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    X, y, n_raw, n_unique = load_data(args.data)
    print(f"Rows: {n_raw} raw -> {n_unique} after removing duplicates")
    print(f"Phishing share: {y.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, m in models.items():
        cv_f1 = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="f1")
        m.fit(X_tr, y_tr)
        fitted[name] = m
        r = evaluate(m, X_te, y_te)
        r.update(model=name, cv_f1_mean=cv_f1.mean(), cv_f1_std=cv_f1.std())
        rows.append(r)
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "model"))

    table = pd.DataFrame(rows).set_index("model").round(4)
    table.to_csv(out / "metrics_table.csv")

    best_name = table["f1"].idxmax()
    best = fitted[best_name]
    print(f"Best model on test F1: {best_name}")

    ConfusionMatrixDisplay.from_estimator(
        best, X_te, y_te, display_labels=["Legitimate", "Phishing"], cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.head(12)[::-1].plot.barh(figsize=(6, 4.5))
    plt.title("Top 12 features (Random Forest)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    plt.close()

    rob = robustness_test(rf, X_te, y_te, list(imp.index[:8]), out / "robustness.png")
    print("Robustness (k forged features -> phishing recall):",
          {k: round(v, 3) for k, v in rob.items()})

    joblib.dump(rf, out / "model.joblib")
    summary = {
        "rows_raw": n_raw, "rows_unique": n_unique,
        "best_model": best_name,
        "metrics": table.to_dict(orient="index"),
        "top_features": imp.head(8).round(4).to_dict(),
        "robustness_recall_by_k": rob,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("Done. Results saved in", out)


if __name__ == "__main__":
    main()
