# models_raw/cnn1d_raw.py
import tensorflow as tf

def build_cnn1d_raw(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),

        # Block 1
        tf.keras.layers.Conv1D(64, kernel_size=5, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Block 2
        tf.keras.layers.Conv1D(128, kernel_size=5, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Block 3
        tf.keras.layers.Conv1D(256, kernel_size=3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalMaxPooling1D(),

        # Dense
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# src/models_raw/cnn1d_raw.py
# import tensorflow as tf

# def build_cnn1d_raw(input_dim, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
#         # Block 1
#         tf.keras.layers.Conv1D(64, 5, padding="same"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.ReLU(),
#         tf.keras.layers.MaxPool1D(2),

#         # Block 2
#         tf.keras.layers.Conv1D(128, 5, padding="same"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.ReLU(),
#         tf.keras.layers.MaxPool1D(2),

#         # Block 3
#         tf.keras.layers.Conv1D(256, 3, padding="same"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.ReLU(),
#         tf.keras.layers.GlobalMaxPool1D(),

#         tf.keras.layers.Dense(256, activation="relu"),
#         tf.keras.layers.Dropout(0.4),
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dropout(0.3),

#         tf.keras.layers.Dense(num_classes, activation="softmax")
#     ])

#     return model

