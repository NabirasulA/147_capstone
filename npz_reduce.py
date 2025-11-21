import numpy as np
from sklearn.model_selection import train_test_split

# Load full 22M binary dataset
npz = np.load("veremi_binary.npz")
X = npz["X"]
y = npz["y"]

print("Loaded:", X.shape, y.shape)

# ================================
# Reduce to 1 Million (balanced)
# ================================
TARGET_SIZE = 1_000_000   # change to 2M or 3M if you want

X_small, _, y_small, _ = train_test_split(
    X, y,
    train_size=TARGET_SIZE,
    random_state=42,
    stratify=y
)

print("Reduced dataset:", X_small.shape, y_small.shape)

np.savez("veremi_binary_1m.npz", X=X_small, y=y_small)
print("Saved veremi_binary_1m.npz")
