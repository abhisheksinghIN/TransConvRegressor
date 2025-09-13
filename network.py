import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, Conv1DTranspose, Dense, Input, Flatten, Dropout, Add,
    LayerNormalization, AveragePooling1D, Concatenate, Cropping1D,
    BatchNormalization, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# ----------------- Custom Functions -----------------
def bounded_relu(x):
    """Activation bounded to [0, 8]."""
    return 8 * K.relu(x) / (1 + K.relu(x))


def transformer_block(x, num_heads, key_dim, ff_dim=128, rate=0.1):
    """Transformer encoder block with residuals and FFN."""
    input_x = x
    x = LayerNormalization()(x)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, attention_axes=1
    )(x, x)
    x = Dropout(rate)(x)
    x = Add()([input_x, x])

    ff_x = LayerNormalization()(x)
    ff_x = Dense(ff_dim, activation="gelu")(ff_x)
    ff_x = Dropout(rate)(ff_x)
    ff_x = Dense(x.shape[-1])(ff_x)
    ff_x = Dropout(rate)(ff_x)

    return Add()([x, ff_x])


def cross_attention_block(query, key, value, num_heads=4, key_dim=32, rate=0.1):
    """Cross-attention block."""
    input_query = query
    x = LayerNormalization()(query)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim
    )(x, key, value)
    x = Dropout(rate)(x)
    return Add()([input_query, x])


def crop_to_match(a, b):
    """Crop tensor a to match shape of b along time axis."""
    diff = a.shape[1] - b.shape[1]
    if diff > 0:
        a = Cropping1D((0, diff))(a)
    elif diff < 0:
        b = Cropping1D((0, -diff))(b)
    return a, b


# ----------------- Model -----------------
def TransConvRegressor(input_shape):
    """1D TransConvRegressor for regression."""
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv1D(64, 3, padding="same", activation="gelu")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = transformer_block(c1, num_heads=2, key_dim=16, ff_dim=64)
    p1 = AveragePooling1D(pool_size=2, padding="same")(c1)

    c2 = Conv1D(128, 3, padding="same", activation="gelu")(p1)
    c2 = BatchNormalization()(c2)
    c2 = transformer_block(c2, num_heads=2, key_dim=32, ff_dim=128)
    p2 = AveragePooling1D(pool_size=2, padding="same")(c2)

    c3 = Conv1D(256, 3, padding="same", activation="gelu")(p2)
    c3 = BatchNormalization()(c3)
    c3 = transformer_block(c3, num_heads=2, key_dim=64, ff_dim=256)
    p3 = AveragePooling1D(pool_size=2, padding="same")(c3)

    c4 = Conv1D(512, 3, padding="same", activation="gelu")(p3)
    c4 = BatchNormalization()(c4)
    c4 = transformer_block(c4, num_heads=2, key_dim=64, ff_dim=512)
    p4 = AveragePooling1D(pool_size=2, padding="same")(c4)

    # Bottleneck
    x = p4
    for _ in range(2):
        x = transformer_block(x, num_heads=4, key_dim=32, ff_dim=512)
    t1 = x

    # Decoder
    u4 = Conv1DTranspose(512, kernel_size=2, strides=2, padding="same")(t1)
    u4 = cross_attention_block(u4, c4, c4)
    u4, c3 = crop_to_match(u4, c4)
    u4 = Concatenate()([u4, c4])
    u4 = Conv1D(512, 3, padding="same", activation="gelu")(u4)
    u4 = BatchNormalization()(u4)

    u3 = Conv1DTranspose(256, 2, strides=2, padding="same")(u4)
    u3 = cross_attention_block(u3, c3, c3)
    u3, c3 = crop_to_match(u3, c3)
    u3 = Concatenate()([u3, c3])
    u3 = Conv1D(256, 3, padding="same", activation="gelu")(u3)
    u3 = BatchNormalization()(u3)

    u2 = Conv1DTranspose(128, 2, strides=2, padding="same")(u3)
    u2 = cross_attention_block(u2, c2, c2)
    u2, c2 = crop_to_match(u2, c2)
    u2 = Concatenate()([u2, c2])
    u2 = Conv1D(128, 3, padding="same", activation="gelu")(u2)
    u2 = BatchNormalization()(u2)

    u1 = Conv1DTranspose(64, 2, strides=2, padding="same")(u2)
    u1 = cross_attention_block(u1, c1, c1)
    u1, c1 = crop_to_match(u1, c1)
    u1 = Concatenate()([u1, c1])
    u1 = Conv1D(64, 3, padding="same", activation="gelu")(u1)
    u1 = BatchNormalization()(u1)

    # Output
    u1 = Flatten()(u1)
    outputs = Dense(1)(u1)
    outputs = Lambda(bounded_relu)(outputs)

    return Model(inputs, outputs)


# ----------------- Custom Loss -----------------
def custom_loss(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    rmse = tf.sqrt(tf.keras.losses.MeanSquaredError()(y_true, y_pred))

    return 0.6 * mae + 0.4 * rmse
