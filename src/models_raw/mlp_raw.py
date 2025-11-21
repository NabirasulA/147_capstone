# models_raw/mlp_raw.py
import tensorflow as tf

def build_mlp_raw(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# # src/models_raw/mlp_raw.py
# import tensorflow as tf

# def build_mlp_raw(input_dim, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),
#         tf.keras.layers.Dense(512, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),

#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.3),

#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.2),

#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     return model
