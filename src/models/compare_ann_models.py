
import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score
)

BASE = "data/processed/normalized_windowed_dataset"
REPORT = "reports/metrics"

os.makedirs(REPORT, exist_ok=True)


def load_data(split):

    files = sorted(
        glob.glob(os.path.join(BASE, split, "*.npz"))
    )

    X_list = []
    y_list = []

    for file in files:

        with np.load(file) as data:
            X_list.append(data["X"])
            y_list.append(data["y30"])

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)

    X = X.reshape(len(X), -1)

    return X, y


X_val, y_val = load_data("validation")

models = {
    "ANN_V1": "models/ann_mlp_best.joblib",
    "ANN_V2": "models/ann_v2_best.joblib"
}

results = []
threshold_results = []

for name, path in models.items():

    model = joblib.load(path)

    probabilities = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, probabilities)
    ap = average_precision_score(y_val, probabilities)
    brier = brier_score_loss(y_val, probabilities)

    prevalence = np.mean(y_val)

    baseline_brier = np.mean(
        (y_val - prevalence) ** 2
    )

    results.append({
        "model": name,
        "validation_auroc": auc,
        "validation_auprc": ap,
        "validation_brier": brier,
        "prevalence_baseline_brier": baseline_brier
    })

    for threshold in np.arange(0.10, 0.91, 0.05):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        threshold_results.append({

            "model": name,
            "threshold": round(float(threshold), 2),

            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),

            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),

            "f1": f1_score(
                y_val, predictions, zero_division=0
            )

        })


results_df = pd.DataFrame(results)
threshold_df = pd.DataFrame(threshold_results)

print("\nVALIDATION MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nVALIDATION THRESHOLD ANALYSIS")
print(threshold_df.to_string(index=False))

results_df.to_csv(
    os.path.join(REPORT, "ann_validation_comparison.csv"),
    index=False
)

threshold_df.to_csv(
    os.path.join(REPORT, "ann_validation_thresholds.csv"),
    index=False
)

print("\nComparison completed.")
