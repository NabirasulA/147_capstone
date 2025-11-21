# main_improved.py
"""
Main training script supporting:
 - improved MLP
 - improved CNN1D
 - XGBoost classifier

Usage examples:
 python main_improved.py --model mlp --cache_npz veremi_binary_1m.npz
 python main_improved.py --model cnn --cache_npz veremi_binary_1m.npz
 python main_improved.py --model xgb --cache_npz veremi_binary_1m.npz
"""

import os, sys, argparse, logging
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# optional: xgboost
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# fix src path if needed (your project uses src/)
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

# import model builders (expected at src/models_raw/*.py or adjust imports)
from models_raw.mlp_raw import build_mlp_raw
from models_raw.cnn1d_raw import build_cnn1d_raw

# SHAP explainer (optional)
from explainability.shap_explainer import SHAPExplainer
from evaluation.metrics import calculate_classification_metrics

# logging
os.makedirs("results/logs", exist_ok=True)
logging.basicConfig(
    filename="results/logs/training.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        logger.info(f"Using GPU: {gpus}")
        print("[INFO] GPU available:", gpus)
    else:
        logger.info("Using CPU")
        print("[INFO] Using CPU")


def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    npz = np.load(path)
    X, y = npz["X"], npz["y"]
    logger.info(f"Loaded NPZ: X={X.shape}, y={y.shape}")
    print(f"[INFO] Loaded NPZ: X={X.shape}, y={y.shape}")
    return X, y


def plot_confusion(y_true, y_pred, title, outpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix {outpath}")
    print(f"[INFO] Saved {outpath}")


def train_nn(model_name, X_train, X_test, y_train, y_test, class_weights, input_dim, num_classes):
    if model_name == "mlp":
        model = build_mlp_raw(input_dim=input_dim, num_classes=num_classes)
        lr = 8e-4
        epochs = 20
        batch_size = 512
    else:  # cnn
        model = build_cnn1d_raw(input_dim=input_dim, num_classes=num_classes)
        lr = 5e-4
        epochs = 20
        batch_size = 512

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    if model_name == "cnn":
        X_train_model = np.expand_dims(X_train, -1)
        X_test_model = np.expand_dims(X_test, -1)
    else:
        X_train_model, X_test_model = X_train, X_test

    history = model.fit(
        X_train_model, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=1
    )

    os.makedirs("results/models", exist_ok=True)
    model_path = f"results/models/{model_name}_model.keras"
    model.save(model_path)
    logger.info(f"Saved model {model_path}")

    preds_proba = model.predict(X_test_model)
    y_pred = np.argmax(preds_proba, axis=1)

    # Metrics + confusion matrix
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n[{model_name.upper()} classification report]")
    print(classification_report(y_test, y_pred))
    logger.info(f"{model_name} report: {report}")

    plot_confusion(y_test, y_pred, f"{model_name.upper()} Confusion Matrix", f"results/confusion_matrices/{model_name}_cm.png")

    return model, report


def train_xgboost(X_train, X_test, y_train, y_test, class_weights):
    if not _HAS_XGB:
        raise RuntimeError("xgboost is not installed in this env. `pip install xgboost`")

    # scikit-learn interface for convenience
    from sklearn.utils import compute_sample_weight
    sample_weights = compute_sample_weight(class_weights, y_train) if class_weights else None

    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=1
    )

    clf.fit(X_train, y_train, sample_weight=sample_weights)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))
    logger.info(f"XGB report: {report}")

    plot_confusion(y_test, preds, "XGBoost Confusion Matrix", "results/confusion_matrices/xgb_cm.png")

    # save model
    os.makedirs("results/models", exist_ok=True)
    clf.save_model("results/models/xgb_model.json")
    logger.info("Saved XGBoost model results/models/xgb_model.json")

    return clf, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn", "xgb"], required=True)
    parser.add_argument("--cache_npz", required=True)
    args = parser.parse_args()

    setup_gpu()
    X, y = load_npz(args.cache_npz)

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # save scaler if you like:
    # joblib.dump(scaler, "results/scaler.joblib")

    unique = np.unique(y)
    num_classes = len(unique)

    # compute class weights (balanced)
    cw_arr = compute_class_weight("balanced", classes=unique, y=y)
    class_weights = {i: w for i, w in enumerate(cw_arr)}
    logger.info(f"class_weights: {class_weights}")
    print("[INFO] class_weights:", class_weights)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    if args.model in ("mlp", "cnn"):
        model, report = train_nn(args.model, X_train, X_test, y_train, y_test, class_weights, input_dim=X.shape[1], num_classes=num_classes)
        # optional: SHAP for MLP only (slow)
        if args.model == "mlp":
            print("[INFO] Computing SHAP (small subset)...")
            shap_exp = SHAPExplainer(model, model_type="kernel")
            X_sub = X_test[:200]
            shap_exp.setup_explainer(X_sub[:20])
            shap_vals = shap_exp.explain_dataset(X_sub)
            fig = shap_exp.plot_summary(shap_vals, X_sub)
            os.makedirs("results", exist_ok=True)
            fig.savefig("results/shap_mlp.png")
            print("[INFO] Saved SHAP -> results/shap_mlp.png")
    else:
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed. pip install xgboost")
        clf, report = train_xgboost(X_train, X_test, y_train, y_test, class_weights)
        # XGBoost SHAP (fast tree explainer)
        try:
            import shap
            explainer = shap.TreeExplainer(clf)
            X_sub = X_test[:200]
            shap_vals = explainer.shap_values(X_sub)
            shap.summary_plot(shap_vals, X_sub, show=False)
            plt.savefig("results/shap_xgb.png", bbox_inches="tight", dpi=200)
            plt.close()
            print("[INFO] Saved SHAP (XGB) -> results/shap_xgb.png")
        except Exception as e:
            logger.warning(f"SHAP for XGB not done: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()
