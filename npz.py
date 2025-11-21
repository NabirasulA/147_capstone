import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_csv("veremi_dataset.csv")

# Remove useless column
df = df.drop(columns=["Unnamed: 0"])

# Binary labels
y_binary = df["attack"].values  # 0 = normal, 1 = attack

# Features (all except attack labels)
X = df.drop(columns=["attack", "attack_type"]).values

print("X shape:", X.shape)
print("y_binary counts:", np.bincount(y_binary))

np.savez("veremi_binary.npz", X=X, y=y_binary)

print("Saved veremi_binary.npz")
