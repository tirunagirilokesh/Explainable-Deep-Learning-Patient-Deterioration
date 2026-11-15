
import os
import glob
import json
import numpy as np
import pandas as pd
import joblib

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
    log_loss
)

# =====================================================
# CONFIGURATION
# =====================================================

SEED = 42

BASE_DIR = "data/processed/normalized_windowed_dataset"

MODEL_DIR = "models"
REPORT_DIR = "reports/metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MAX_EPOCHS = 100
PATIENCE = 12
THRESHOLD = 0.50


# =====================================================
# LOAD DATA
# =====================================================

def load_split(split):

    folder = os.path.join(BASE_DIR, split)

    files = sorted(
        glob.glob(os.path.join(folder, "*.npz"))
    )

    if not files:
        raise FileNotFoundError(
            f"No NPZ files found in {folder}"
        )

    X_list = []
    y_list = []

    for file in files:

        with np.load(file) as data:

            X = data["X"]
            y = data["y30"]

            if X.ndim != 3 or X.shape[1:] != (900, 5):
                raise ValueError(
                    f"Unexpected X shape in {file}: {X.shape}"
                )

            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Flatten temporal windows for ANN
    X = X.reshape(X.shape[0], -1)

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite values found in {split} features."
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"Unexpected target labels in {split}."
        )

    print(f"\n{split.upper()} SUMMARY")
    print("-" * 40)
    print("Patients with files:", len(files))
    print("Windows:", len(y))
    print("Input shape:", X.shape)
    print("Positive:", int(np.sum(y == 1)))
    print("Negative:", int(np.sum(y == 0)))

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("validation")
X_test, y_test = load_split("test")


# =====================================================
# CLASS BALANCING
# =====================================================

negative = np.sum(y_train == 0)
positive = np.sum(y_train == 1)

if negative == 0 or positive == 0:
    raise ValueError("Training data must contain both classes.")

n_train = len(y_train)

weight_negative = n_train / (2 * negative)
weight_positive = n_train / (2 * positive)

sample_weights = np.where(
    y_train == 1,
    weight_positive,
    weight_negative
).astype(np.float64)

print("\nCLASS WEIGHTS")
print("Negative:", weight_negative)
print("Positive:", weight_positive)


# =====================================================
# ANN MODEL
# =====================================================

model = MLPClassifier(

    hidden_layer_sizes=(256, 128, 64),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    batch_size=128,

    alpha=0.0001,

    max_iter=1,

    shuffle=True,

    random_state=SEED,

    warm_start=False
)


# =====================================================
# TRAINING WITH VALIDATION MONITORING
# =====================================================

best_ap = -1
best_epoch = 0
wait = 0

history = []

best_model_path = os.path.join(
    MODEL_DIR,
    "ann_mlp_best.joblib"
)

print("\nSTARTING ANN TRAINING")
print("=" * 60)

for epoch in range(1, MAX_EPOCHS + 1):

    # One incremental training epoch
    model.partial_fit(
        X_train,
        y_train,
        classes=np.array([0, 1]),
        sample_weight=sample_weights
    )

    # Validation predictions
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

    history.append({

        "epoch": epoch,
        "validation_loss": val_loss,
        "validation_auprc": val_ap,
        "validation_auroc": val_auc

    })

    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val AUPRC: {val_ap:.4f} | "
        f"Val AUROC: {val_auc:.4f}"
    )

    # Select checkpoint using validation AUPRC
    if val_ap > best_ap + 0.0001:

        best_ap = val_ap
        best_epoch = epoch
        wait = 0

        joblib.dump(model, best_model_path)

        print("  Best validation AUPRC - model saved")

    else:

        wait += 1

    if wait >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =====================================================
# LOAD BEST VALIDATION MODEL
# =====================================================

best_model = joblib.load(best_model_path)

print("\nBest epoch:", best_epoch)
print("Best validation AUPRC:", round(best_ap, 4))


# =====================================================
# FINAL TEST EVALUATION
# =====================================================

y_prob = best_model.predict_proba(X_test)[:, 1]

# Baseline threshold only
y_pred = (y_prob >= THRESHOLD).astype(int)

metrics = {

    "model": "ANN_MLPClassifier",

    "best_epoch": int(best_epoch),

    "threshold": THRESHOLD,

    "accuracy": float(
        accuracy_score(y_test, y_pred)
    ),

    "precision": float(
        precision_score(
            y_test, y_pred, zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            y_test, y_pred, zero_division=0
        )
    ),

    "f1_score": float(
        f1_score(
            y_test, y_pred, zero_division=0
        )
    ),

    "roc_auc": float(
        roc_auc_score(y_test, y_prob)
    ),

    "pr_auc": float(
        average_precision_score(y_test, y_prob)
    )

}


print("\nANN TEST RESULTS")
print("=" * 60)

for key, value in metrics.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# =====================================================
# CONFUSION MATRIX AND CLASSIFICATION REPORT
# =====================================================

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
    target_names=[
        "No Hypotension",
        "Hypotension"
    ],
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No Hypotension",
            "Hypotension"
        ],
        zero_division=0
    )
)


# =====================================================
# SAVE RESULTS
# =====================================================

with open(
    os.path.join(REPORT_DIR, "ann_mlp_metrics.json"),
    "w"
) as f:

    json.dump(metrics, f, indent=4)


pd.DataFrame(history).to_csv(
    os.path.join(REPORT_DIR, "ann_mlp_training_history.csv"),
    index=False
)

pd.DataFrame(
    cm,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"]
).to_csv(
    os.path.join(REPORT_DIR, "ann_mlp_confusion_matrix.csv")
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(REPORT_DIR, "ann_mlp_classification_report.csv")
)

np.savez_compressed(
    os.path.join(REPORT_DIR, "ann_mlp_test_predictions.npz"),
    y_true=y_test,
    y_probability=y_prob,
    y_predicted=y_pred
)

print("\nANN TRAINING COMPLETED")
print("Best model:", best_model_path)
print("Reports:", REPORT_DIR)
