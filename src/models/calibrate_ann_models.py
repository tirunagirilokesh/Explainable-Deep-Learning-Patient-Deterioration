
import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss
)

BASE = "data/processed/normalized_windowed_dataset"
REPORT = "reports/metrics"

os.makedirs(REPORT, exist_ok=True)


def load_split(split):

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

    return X.reshape(len(X), -1), y


X_val, y_val = load_split("validation")

# Original training class weights
w0 = 0.5972290489063652
w1 = 3.071248025276461

models = {
    "ANN_V1": "models/ann_mlp_best.joblib",
    "ANN_V2": "models/ann_v2_best.joblib"
}

results = []

for name, path in models.items():

    model = joblib.load(path)

    p = model.predict_proba(X_val)[:, 1]

    # Correct probabilities for the training class weights
    corrected = (p / w1) / (
        (p / w1) + ((1 - p) / w0)
    )

    for method, probabilities in [
        ("Original", p),
        ("Weight_corrected", corrected)
    ]:

        probabilities = np.clip(
            probabilities, 1e-7, 1 - 1e-7
        )

        results.append({

            "model": name,
            "method": method,

            "AUROC": roc_auc_score(
                y_val, probabilities
            ),

            "AUPRC": average_precision_score(
                y_val, probabilities
            ),

            "Brier": brier_score_loss(
                y_val, probabilities
            ),

            "Log_loss": log_loss(
                y_val, probabilities
            )

        })


df = pd.DataFrame(results)

print("\nVALIDATION CALIBRATION COMPARISON")
print(df.to_string(index=False))

df.to_csv(
    os.path.join(
        REPORT,
        "ann_weight_correction_comparison.csv"
    ),
    index=False
)

print("\nResults saved.")
