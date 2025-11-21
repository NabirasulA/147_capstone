import numpy as np
import tensorflow as tf
from src.explainability.lime_explainer import LIMEExplainer

# Load dataset
npz = np.load("veremi_binary_1m.npz")
X = npz["X"]       # shape = (N, 18)
y = npz["y"]

# CNN expects (18,1)
X_cnn = X.reshape(X.shape[0], X.shape[1], 1)

# Load trained CNN model
model = tf.keras.models.load_model("results/models/cnn1d_raw.keras")

# Feature names
feature_names = [
    "type","rcvTime","pos_0","pos_1","pos_noise_0","pos_noise_1",
    "spd_0","spd_1","spd_noise_0","spd_noise_1","acl_0","acl_1",
    "acl_noise_0","acl_noise_1","hed_0","hed_1","hed_noise_0","hed_noise_1"
]

class_names = ["normal", "attack"]

# Create LIME Explainer
lime = LIMEExplainer(
    model=model,
    feature_names=feature_names,
    class_names=class_names
)

# IMPORTANT: LIME must be trained on TABULAR input, not CNN reshaped input
lime.setup_explainer(X)        # <-- FIXED

# Explain one instance
fig = lime.explain_instance(X[0])
fig.savefig("lime_output.png")

print("Saved lime_output.png")
