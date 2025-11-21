import numpy as np
import tensorflow as tf
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -----------------------------------------------------------------------------
# Load dataset from your existing cache
# -----------------------------------------------------------------------------
data = np.load("veremi_cached.npz")
X = data["X"]
y = data["y"]

print("Loaded X:", X.shape)
print("Loaded y:", y.shape)

# -----------------------------------------------------------------------------
# Normalize features (must be same as training)
# -----------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------------
# Load your trained models (update paths if needed)
# -----------------------------------------------------------------------------
mlp = tf.keras.models.load_model("results/models/mlp_raw.keras")
cnn = tf.keras.models.load_model("results/models/cnn_raw.keras")

print("\nLoaded Models:")
print(" - MLP:", mlp)
print(" - CNN:", cnn)

# -----------------------------------------------------------------------------
# Create output folder for plots
# -----------------------------------------------------------------------------
os.makedirs("results/plots", exist_ok=True)

# -----------------------------------------------------------------------------
# SHAP for MLP
# -----------------------------------------------------------------------------
print("\nRunning SHAP for MLP...")

# Select background samples (SHAP recommends small number)
background = X_scaled[np.random.choice(X_scaled.shape[0], 200, replace=False)]

explainer_mlp = shap.KernelExplainer(mlp.predict, background)

# Pick 20 samples to explain
sample = X_scaled[np.random.choice(X_scaled.shape[0], 20, replace=False)]

shap_values_mlp = explainer_mlp.shap_values(sample)

# Plot SHAP summary
plt.figure()
shap.summary_plot(shap_values_mlp, sample, show=False)
plt.savefig("results/plots/shap_summary_mlp.png", dpi=200, bbox_inches='tight')
plt.close()

print("✔ Saved SHAP summary for MLP → results/plots/shap_summary_mlp.png")

# -----------------------------------------------------------------------------
# SHAP for CNN-1D
# -----------------------------------------------------------------------------
print("\nRunning SHAP for CNN-1D...")

# CNN input needs reshaping
X_cnn = X_scaled.reshape(-1, 20, 1)
background_cnn = X_cnn[np.random.choice(X_cnn.shape[0], 200, replace=False)]
sample_cnn = X_cnn[np.random.choice(X_cnn.shape[0], 20, replace=False)]

explainer_cnn = shap.KernelExplainer(
    lambda x: cnn.predict(x.reshape(-1, 20, 1)),
    background_cnn.reshape(-1, 20)
)

shap_values_cnn = explainer_cnn.shap_values(sample_cnn.reshape(-1, 20))

# Plot SHAP summary
plt.figure()
shap.summary_plot(shap_values_cnn, sample_cnn.reshape(-1, 20), show=False)
plt.savefig("results/plots/shap_summary_cnn.png", dpi=200, bbox_inches='tight')
plt.close()

print("✔ Saved SHAP summary for CNN → results/plots/shap_summary_cnn.png")

# -----------------------------------------------------------------------------
# LIME for tabular data (MLP and CNN both use tabular inputs)
# -----------------------------------------------------------------------------
print("\nRunning LIME...")

class_names = [str(c) for c in np.unique(y)]

lime_explainer = LimeTabularExplainer(
    X_scaled,
    feature_names=[f"F{i+1}" for i in range(X_scaled.shape[1])],
    class_names=class_names,
    mode="classification"
)

# Pick 1 instance to explain
idx = np.random.randint(0, X_scaled.shape[0])
instance = X_scaled[idx]

# Explain with MLP
exp_mlp = lime_explainer.explain_instance(
    instance,
    mlp.predict,
    num_features=10
)

exp_mlp.save_to_file("results/plots/lime_mlp.html")
print("✔ Saved LIME (MLP) → results/plots/lime_mlp.html")

# Explain with CNN
exp_cnn = lime_explainer.explain_instance(
    instance,
    lambda x: cnn.predict(x.reshape(-1, 20, 1)),
    num_features=10
)

exp_cnn.save_to_file("results/plots/lime_cnn.html")
print("✔ Saved LIME (CNN) → results/plots/lime_cnn.html")

print("\n🎉 SHAP + LIME explanations completed successfully!")
print("Check results/plots/ for images and HTML files.")
