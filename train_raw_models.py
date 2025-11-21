import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

npz = np.load("veremi_cached.npz")
X = npz["X"]     # shape (1188280, 20)
y = npz["y"]

print("Loaded:")
print("X:", X.shape)
print("y:", y.shape)

# -------------------------------------------------------------------
# Normalization (very important for MLP/CNN)
# -------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------------------------------------------
# Train-test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------------------------------------------
# MLP MODEL (Recommended)
# -------------------------------------------------------------------
mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

mlp.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining MLP...")
mlp_history = mlp.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=1024,
    verbose=1
)

# -------------------------------------------------------------------
# CNN-1D MODEL (Optional, usually similar performance)
# -------------------------------------------------------------------
X_train_cnn = X_train.reshape(-1, 20, 1)
X_test_cnn  = X_test.reshape(-1, 20, 1)

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(20, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN-1D...")
cnn_history = cnn.fit(
    X_train_cnn, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=1024,
    verbose=1
)

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
print("\nEvaluating MLP...")
mlp_eval = mlp.evaluate(X_test, y_test)

print("\nEvaluating CNN-1D...")
cnn_eval = cnn.evaluate(X_test_cnn, y_test)

# -------------------------------------------------------------------
# Save models
# -------------------------------------------------------------------
mlp.save("results/models/mlp_raw.keras")
cnn.save("results/models/cnn_raw.keras")

print("\nSaved models:")
print(" - results/models/mlp_raw.keras")
print(" - results/models/cnn_raw.keras")
