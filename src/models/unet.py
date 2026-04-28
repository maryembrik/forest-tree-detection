"""
Improved U-Net with EfficientNetB0 encoder, attention gates, BN, and combined loss.
Targets Dice > 0.75 on dead tree binary segmentation.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import tensorflow.keras.backend as K


# ── Loss functions ────────────────────────────────────────────────────────────

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Soft Dice coefficient (used both as metric and inside loss)."""
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Binary focal loss with class-balancing alpha."""
    eps = 1e-8
    y_pred = K.clip(y_pred, eps, 1.0 - eps)
    y_true = tf.cast(y_true, tf.float32)
    p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
    focal = -alpha_t * K.pow(1.0 - p_t, gamma) * K.log(p_t)
    return K.mean(focal)


def combined_loss(dice_w=0.5, focal_w=0.5):
    """Factory returning a combined Dice + Focal loss function."""
    def loss(y_true, y_pred):
        return dice_w * dice_loss(y_true, y_pred) + focal_w * binary_focal_loss(y_true, y_pred)
    loss.__name__ = "combined_loss"
    return loss


# ── Architectural building blocks ─────────────────────────────────────────────

def attention_gate(x, g, inter_channels):
    """
    Soft attention gate applied to a skip connection tensor `x` using
    the gating signal `g` from the decoder path.

    Args:
        x: Skip-connection feature map (from encoder).
        g: Gating signal (from decoder, coarser resolution).
        inter_channels: Number of intermediate channels for attention computation.

    Returns:
        Attention-weighted version of `x`.
    """
    theta_x = layers.Conv2D(inter_channels, 1, padding='same')(x)
    phi_g   = layers.Conv2D(inter_channels, 1, padding='same')(g)

    # Upsample gating signal if spatial dims differ
    x_shape = tf.shape(theta_x)
    g_shape = tf.shape(phi_g)
    phi_g = tf.image.resize(phi_g, [x_shape[1], x_shape[2]], method='bilinear')

    add     = layers.Add()([theta_x, phi_g])
    relu    = layers.Activation('relu')(add)
    psi     = layers.Conv2D(1, 1, padding='same')(relu)
    sigmoid = layers.Activation('sigmoid')(psi)
    return layers.Multiply()([x, sigmoid])


def conv_bn_relu(x, filters, kernel_size=3):
    """Conv2D → BatchNorm → ReLU block."""
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def decoder_block(x, skip, filters, dropout=0.0):
    """
    One decoder stage: transpose-conv upsample + optional attention skip + two Conv-BN-ReLU.

    Args:
        x:       Incoming tensor from previous (coarser) decoder stage.
        skip:    Skip-connection tensor from encoder (or None if no skip).
        filters: Number of output feature-map channels.
        dropout: Dropout rate applied after convolutions (0 = disabled).

    Returns:
        Upsampled and merged feature map.
    """
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    if skip is not None:
        skip_att = attention_gate(skip, x, max(filters // 2, 1))
        x = layers.Concatenate()([x, skip_att])
    x = conv_bn_relu(x, filters)
    x = conv_bn_relu(x, filters)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    return x


# ── Model builder ─────────────────────────────────────────────────────────────

def build_efficientnet_unet(
    input_shape=(224, 224, 3),
    dropout_rate=0.3,
    freeze_encoder=False,
):
    """
    U-Net with EfficientNetB0 encoder (pretrained on ImageNet).

    Skip connections are tapped at four resolution levels:
        s1  →  112 × 112  (stem_activation)
        s2  →   56 ×  56  (block2a_expand_activation)
        s3  →   28 ×  28  (block3a_expand_activation)
        s4  →   14 ×  14  (block4a_expand_activation)
    Bridge: 7 × 7  (top_activation)

    Args:
        input_shape:    HWC tuple (default 224×224×3).
        dropout_rate:   Dropout applied at the bottleneck.
        freeze_encoder: When True, encoder weights are non-trainable.

    Returns:
        Keras Model with sigmoid output of shape (H, W, 1).
    """
    inputs = layers.Input(shape=input_shape)

    # ── Encoder ──────────────────────────────────────────────────────────────
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
    )

    if freeze_encoder:
        for layer in base.layers:
            layer.trainable = False

    # Helper: find last layer whose name contains a substring
    def _get(substr):
        matches = [l for l in base.layers if substr in l.name]
        if not matches:
            return None
        return matches[-1].output

    # EfficientNetB0 skip-connection layer names (224×224 input)
    s1     = _get('stem_activation')             # 112 × 112 ×  32
    s2     = _get('block2a_expand_activation')   #  56 ×  56 ×  96
    s3     = _get('block3a_expand_activation')   #  28 ×  28 × 144
    s4     = _get('block4a_expand_activation')   #  14 ×  14 × 240
    bridge = _get('top_activation')              #   7 ×   7 × 1280

    # Fallback: use model output if top_activation not found
    if bridge is None:
        bridge = base.output

    # ── Bottleneck dropout ────────────────────────────────────────────────────
    x = layers.Dropout(dropout_rate)(bridge)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = decoder_block(x, s4, 256)   #  7 →  14
    x = decoder_block(x, s3, 128)   # 14 →  28
    x = decoder_block(x, s2,  64)   # 28 →  56
    x = decoder_block(x, s1,  32)   # 56 → 112
    x = decoder_block(x, None, 16)  # 112 → 224  (no skip at full resolution)

    # ── Output head ──────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name='output')(x)

    model = models.Model(
        inputs=[inputs],
        outputs=[outputs],
        name="EfficientNetUNet",
    )
    return model


# ── Compilation helper ────────────────────────────────────────────────────────

def compile_model(model, lr=1e-3, weight_decay=1e-4):
    """
    Compile the model with AdamW (falls back to Adam if unavailable) and
    combined Dice + Focal loss.

    Args:
        model:        Keras Model returned by build_efficientnet_unet().
        lr:           Initial learning rate.
        weight_decay: L2 weight decay for AdamW.

    Returns:
        Compiled model (in-place compile, also returned for convenience).
    """
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )
    except AttributeError:
        # TF < 2.11 fallback
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=combined_loss(dice_w=0.5, focal_w=0.5),
        metrics=[
            dice_coefficient,
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    return model


# ── Training callbacks ────────────────────────────────────────────────────────

def get_callbacks(
    save_path="outputs/best_unet_model.keras",
    patience_lr=5,
    patience_stop=15,
):
    """
    Standard callback set: ModelCheckpoint, ReduceLROnPlateau, EarlyStopping.

    Args:
        save_path:     Path for the best-model checkpoint.
        patience_lr:   Epochs without improvement before reducing LR.
        patience_stop: Epochs without improvement before early stopping.

    Returns:
        List of Keras callbacks.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            verbose=1,
            min_lr=1e-7,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_stop,
            verbose=1,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='outputs/logs',
            histogram_freq=1,
        ),
    ]


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_efficientnet_unet(input_shape=(224, 224, 3))
    model = compile_model(model)
    model.summary()
    print("\nOutput shape:", model.output_shape)   # should be (None, 224, 224, 1)
