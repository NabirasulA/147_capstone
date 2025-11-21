# main.py  (Binary Classification + Scaling + Class Weights +
#           Improved MLP/CNN1D + SHAP + Confusion Matrices + ROC/AUC)
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Logging setup
# -------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
(RESULTS / "logs").mkdir(parents=True, exist_ok=True)
(RESULTS / "models").mkdir(parents=True, exist_ok=True)
(RESULTS / "confusion_matrices").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(RESULTS / "logs" / "training.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vanet")

# -------------------------
# Fix python path for src
# -------------------------
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# -------------------------
# Imports from project
# -------------------------
from models_raw.mlp_raw import build_mlp_raw
from models_raw.cnn1d_raw import build_cnn1d_raw
from evaluation.metrics import calculate_classification_metrics
from explainability.shap_explainer import SHAPExplainer

# -------------------------
# Utility plotting
# -------------------------
def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix", labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_roc(y_true, y_proba, save_path, pos_label=1):
    # y_proba shape: (n_samples, n_classes) or (n_samples,) for binary prob of positive
    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        score = y_proba[:, pos_label]
    else:
        score = y_proba.ravel()
    fpr, tpr, _ = roc_curve(y_true, score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return roc_auc

# -------------------------
# GPU Setup
# -------------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        logger.info(f"Using GPUs: {gpus}")
        print("[INFO] GPUs found - using GPU.")
    else:
        logger.info("No GPUs found - using CPU.")
        print("[INFO] No GPUs - using CPU.")

# -------------------------
# Load dataset
# -------------------------
def load_npz(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(str(path))
    X = data["X"]
    y = data["y"]
    logger.info(f"Loaded NPZ: X={X.shape}, y={y.shape}")
    return X, y

# -------------------------
# Save scaler
# -------------------------
def save_scaler(scaler, out_path):
    import joblib
    joblib.dump(scaler, out_path)
    logger.info(f"Saved scaler at {out_path}")

# -------------------------
# Training function
# -------------------------
def train_raw(cache_npz,
              model_prefix="raw",
              epochs=20,
              batch_size=512,
              save_models=True):

    X, y = load_npz(cache_npz)
    num_classes = int(len(np.unique(y)))
    logger.info(f"Num classes: {num_classes}")

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    save_scaler(scaler, RESULTS / f"{model_prefix}_scaler.joblib")

    # class weights
    cw = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights = {i: cw_i for i, cw_i in enumerate(cw)}
    logger.info(f"Class weights: {class_weights}")
    print("[INFO] Class weights:", class_weights)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ---------------- MLP ----------------
    mlp = build_mlp_raw(input_dim=X.shape[1], num_classes=num_classes)
    mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(RESULTS / "models" / f"mlp_best.keras"),
                                           monitor="val_loss", save_best_only=True)
    ]

    history_mlp = mlp.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    if save_models:
        mlp.save(RESULTS / "models" / "mlp_final.keras")
        logger.info("Saved final MLP model")

    # Eval MLP
    mlp_probs = mlp.predict(X_test, batch_size=1024)
    mlp_pred = np.argmax(mlp_probs, axis=1)
    mlp_metrics = calculate_classification_metrics(y_test, mlp_pred, mlp_probs)
    logger.info(f"MLP Metrics: {mlp_metrics}")
    print("[MLP METRICS]", mlp_metrics)

    plot_confusion_matrix(y_test, mlp_pred, RESULTS / "confusion_matrices" / "mlp_cm.png",
                          title="MLP Confusion Matrix", labels=[0,1] if num_classes==2 else None)
    auc_mlp = plot_roc(y_test, mlp_probs, RESULTS / "confusion_matrices" / "mlp_roc.png")
    logger.info(f"MLP AUC: {auc_mlp:.4f}")

    # ---------------- CNN ----------------
    X_train_cnn = np.expand_dims(X_train, -1)
    X_test_cnn = np.expand_dims(X_test, -1)

    cnn = build_cnn1d_raw(input_dim=X.shape[1], num_classes=num_classes)
    cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(RESULTS / "models" / "cnn_best.keras"),
                                           monitor="val_loss", save_best_only=True)
    ]

    history_cnn = cnn.fit(
        X_train_cnn, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    if save_models:
        cnn.save(RESULTS / "models" / "cnn_final.keras")
        logger.info("Saved final CNN model")

    # Eval CNN
    cnn_probs = cnn.predict(X_test_cnn, batch_size=1024)
    cnn_pred = np.argmax(cnn_probs, axis=1)
    cnn_metrics = calculate_classification_metrics(y_test, cnn_pred, cnn_probs)
    logger.info(f"CNN Metrics: {cnn_metrics}")
    print("[CNN METRICS]", cnn_metrics)

    plot_confusion_matrix(y_test, cnn_pred, RESULTS / "confusion_matrices" / "cnn_cm.png",
                          title="CNN1D Confusion Matrix", labels=[0,1] if num_classes==2 else None)
    auc_cnn = plot_roc(y_test, cnn_probs, RESULTS / "confusion_matrices" / "cnn_roc.png")
    logger.info(f"CNN AUC: {auc_cnn:.4f}")

    # Save metrics JSON
    meta = {
        "mlp_metrics": mlp_metrics,
        "cnn_metrics": cnn_metrics,
        "mlp_auc": float(auc_mlp),
        "cnn_auc": float(auc_cnn),
        "class_weights": class_weights,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(RESULTS / "logs" / "metrics.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return mlp, cnn, X_test, y_test

# -------------------------
# SHAP explainability (MLP)
# -------------------------
def explain_shap(mlp, X_test, n_samples=200):
    X_exp = X_test[:n_samples]
    shap = SHAPExplainer(mlp, model_type="kernel")
    shap.setup_explainer(X_exp[: min(50, len(X_exp)) ])
    shap_values = shap.explain_dataset(X_exp)
    fig = shap.plot_summary(shap_values, X_exp)
    fig.savefig(RESULTS / "shap_mlp.png")
    logger.info("Saved SHAP plot")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_npz", required=True, help="Path to cached .npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    setup_gpu()
    mlp, cnn, X_test, y_test = train_raw(args.cache_npz, epochs=args.epochs, batch_size=args.batch_size)
    explain_shap(mlp, X_test, n_samples=200)
    print("Done. Results in:", RESULTS)

if __name__ == "__main__":
    main()


