
import os
import glob
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    log_loss,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42

BASE_DIR = "data/processed/normalized_windowed_dataset"
MODEL_DIR = "models"
REPORT_DIR = "reports/metrics"

MAX_EPOCHS = 100
PATIENCE = 15
THRESHOLD = 0.50

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

def load_split(split):

    folder = os.path.join(BASE_DIR, split)

    files = sorted(
        glob.glob(os.path.join(folder, "*.npz"))
    )

    if not files:
        raise FileNotFoundError(folder)

    X_list = []
    y_list = []

    for file in files:

        with np.load(file) as data:

            X_list.append(data["X"])
            y_list.append(data["y30"])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    X = X.reshape(X.shape[0], -1)

    if not np.isfinite(X).all():
        raise ValueError(f"Invalid values in {split}")

    print(f"\n{split.upper()}")
    print("Patients:", len(files))
    print("Windows:", len(y))
    print("Features:", X.shape)
    print("Positive:", int(np.sum(y == 1)))
    print("Negative:", int(np.sum(y == 0)))

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")


# =====================================================
# CLASS WEIGHTS
# =====================================================

negative = np.sum(y_train == 0)
positive = np.sum(y_train == 1)

n = len(y_train)

weight_0 = n / (2 * negative)
weight_1 = n / (2 * positive)

sample_weights = np.where(
    y_train == 1,
    weight_1,
    weight_0
).astype(np.float64)

print("\nCLASS WEIGHTS")
print("Class 0:", weight_0)
print("Class 1:", weight_1)


# =====================================================
# ANN V2 MODEL
# =====================================================

model = MLPClassifier(

    hidden_layer_sizes=(128, 64),

    activation="relu",

    solver="adam",

    learning_rate_init=0.0001,

    batch_size=128,

    alpha=0.001,

    max_iter=1,

    shuffle=True,

    random_state=SEED,

    warm_start=False
)


# =====================================================
# TRAINING
# =====================================================

best_ap = -1
best_epoch = 0
wait = 0

history = []

best_path = os.path.join(
    MODEL_DIR,
    "ann_v2_best.joblib"
)

print("\nANN V2 TRAINING")
print("=" * 60)

for epoch in range(1, MAX_EPOCHS + 1):

    model.partial_fit(
        X_train,
        y_train,
        classes=np.array([0, 1]),
        sample_weight=sample_weights
    )

    val_prob = model.predict_proba(X_val)[:, 1]

    val_loss = log_loss(
        y_val,
        val_prob,
        labels=[0, 1]
    )

    val_ap = average_precision_score(
        y_val,
        val_prob
    )

    val_auc = roc_auc_score(
        y_val,
        val_prob
    )

    val_brier = brier_score_loss(
        y_val,
        val_prob
    )

    history.append({
        "epoch": epoch,
        "validation_loss": val_loss,
        "validation_auprc": val_ap,
        "validation_auroc": val_auc,
        "validation_brier": val_brier
    })

    print(
        f"Epoch {epoch:03d} | "
        f"Loss: {val_loss:.4f} | "
        f"AUPRC: {val_ap:.4f} | "
        f"AUROC: {val_auc:.4f} | "
        f"Brier: {val_brier:.4f}"
    )

    if val_ap > best_ap + 0.0001:

        best_ap = val_ap
        best_epoch = epoch
        wait = 0

        joblib.dump(model, best_path)

        print("  Best checkpoint saved.")

    else:

        wait += 1

    if wait >= PATIENCE:

        print("\nEarly stopping.")
        break


# =====================================================
# RESTORE BEST CHECKPOINT
# =====================================================

best_model = joblib.load(best_path)

print("\nBest epoch:", best_epoch)
print("Best validation AUPRC:", best_ap)


# =====================================================
# VALIDATION CALIBRATION
# =====================================================

val_prob = best_model.predict_proba(X_val)[:, 1]

val_brier = brier_score_loss(
    y_val,
    val_prob
)

fraction_positive, mean_probability = calibration_curve(
    y_val,
    val_prob,
    n_bins=10,
    strategy="quantile"
)

calibration_df = pd.DataFrame({
    "mean_predicted_probability": mean_probability,
    "observed_positive_fraction": fraction_positive
})

calibration_df.to_csv(
    os.path.join(REPORT_DIR, "ann_v2_validation_calibration.csv"),
    index=False
)

print("\nVALIDATION CALIBRATION")
print("Brier score:", round(val_brier, 5))


# =====================================================
# TEST EVALUATION
# =====================================================

y_prob = best_model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= THRESHOLD).astype(int)

metrics = {

    "model": "ANN_V2_MLPClassifier",

    "best_epoch": int(best_epoch),

    "threshold": THRESHOLD,

    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),

    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),

    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),

    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),

    "roc_auc": float(
        roc_auc_score(y_test, y_prob)
    ),

    "pr_auc": float(
        average_precision_score(y_test, y_prob)
    ),

    "brier_score": float(
        brier_score_loss(y_test, y_prob)
    )
}


print("\nANN V2 TEST RESULTS")
print("=" * 60)

for key, value in metrics.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print(cm)

print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Hypotension", "Hypotension"],
    zero_division=0
)

print(report)


# =====================================================
# SAVE OUTPUTS
# =====================================================

with open(
    os.path.join(REPORT_DIR, "ann_v2_metrics.json"),
    "w"
) as f:

    json.dump(metrics, f, indent=4)


pd.DataFrame(history).to_csv(
    os.path.join(REPORT_DIR, "ann_v2_training_history.csv"),
    index=False
)

pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
).to_csv(
    os.path.join(REPORT_DIR, "ann_v2_confusion_matrix.csv")
)

np.savez_compressed(
    os.path.join(REPORT_DIR, "ann_v2_test_predictions.npz"),
    y_true=y_test,
    y_probability=y_prob,
    y_predicted=y_pred
)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "ann_v2_final.joblib")
)

print("\nANN V2 COMPLETED")
print("Model:", best_path)
print("Reports:", REPORT_DIR)
