import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess


def _ensure_float01(x: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    return tf.clip_by_value(x, 0.0, 1.0)


def get_vgg_perceptual_model(input_shape):
    """
    入力: (H, W, 3) の [0,1] を想定
    出力: 中間層特徴（block3_conv3, block4_conv3）
    """
    vgg = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    vgg.trainable = False

    outputs = [
        vgg.get_layer("block3_conv3").output,
        vgg.get_layer("block4_conv3").output,
    ]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs, name="vgg_perceptual")
    model.trainable = False
    return model


def perceptual_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    vgg_model: tf.keras.Model,
    mask: tf.Tensor | None = None,
) -> tf.Tensor:
    """
    y_true, y_pred: (B,H,W,3) [0,1]
    mask: (B,H,W,1) 1:hole, 0:context（任意）
    """
    y_true = _ensure_float01(y_true)
    y_pred = _ensure_float01(y_pred)

    yt = vgg_preprocess(y_true * 255.0)
    yp = vgg_preprocess(y_pred * 255.0)

    feats_t = vgg_model(yt)
    feats_p = vgg_model(yp)

    if not isinstance(feats_t, (list, tuple)):
        feats_t = [feats_t]
        feats_p = [feats_p]

    loss = 0.0
    for ft, fp in zip(feats_t, feats_p):
        if mask is not None:
            m = tf.image.resize(mask, tf.shape(ft)[1:3], method="nearest")
            diff = tf.abs(ft - fp)
            loss_layer = tf.reduce_sum(diff * m) / (tf.reduce_sum(m) + 1e-6)
        else:
            loss_layer = tf.reduce_mean(tf.abs(ft - fp))
        loss += loss_layer

    return loss / float(len(feats_t))
