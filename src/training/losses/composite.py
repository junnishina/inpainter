import tensorflow as tf

from .perceptual import perceptual_loss


def _to_float01(y_true: tf.Tensor, y_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    yt = tf.clip_by_value(y_true, 0.0, 1.0)
    yp = tf.clip_by_value(y_pred, 0.0, 1.0)
    return yt, yp


def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    yt, yp = _to_float01(y_true, y_pred)
    ssim = tf.image.ssim(yt, yp, max_val=1.0)
    return 1.0 - tf.reduce_mean(ssim)


def ms_ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    yt, yp = _to_float01(y_true, y_pred)
    msssim = tf.image.ssim_multiscale(yt, yp, max_val=1.0)
    msssim = tf.where(tf.math.is_finite(msssim), msssim, tf.zeros_like(msssim))
    return 1.0 - tf.reduce_mean(msssim)


def gradient_l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    yt, yp = _to_float01(y_true, y_pred)
    sobel_true = tf.image.sobel_edges(yt)
    sobel_pred = tf.image.sobel_edges(yp)
    grad_true = tf.abs(sobel_true[..., 0]) + tf.abs(sobel_true[..., 1])
    grad_pred = tf.abs(sobel_pred[..., 0]) + tf.abs(sobel_pred[..., 1])
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))


def _avg_pool_2x(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.avg_pool2d(x, ksize=2, strides=2, padding="SAME")


def ssim_multiscale_stable(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    value_range=(0.0, 1.0),
    max_levels=3,
    filter_size=7,
    k1=0.05,
    k2=0.05,
    weights=None,
    mask: tf.Tensor | None = None,
) -> tf.Tensor:
    """
    数値安定化した MS-SSIM 風の loss (1 - similarity)
    """
    lo, hi = value_range
    yt = tf.cast(y_true, tf.float32)
    yp = tf.cast(y_pred, tf.float32)

    scale = tf.maximum(hi - lo, 1e-6)
    yt = (yt - lo) / scale
    yp = (yp - lo) / scale

    eps = 1e-6
    yt = tf.clip_by_value(yt, eps, 1.0 - eps)
    yp = tf.clip_by_value(yp, eps, 1.0 - eps)

    tf.debugging.assert_all_finite(yt, "yt_before_ssim has NaN/Inf")
    tf.debugging.assert_all_finite(yp, "yp_before_ssim has NaN/Inf")

    std_pf5 = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    if weights is None:
        w = std_pf5[:max_levels]
        s = sum(w)
        weights = [wi / s for wi in w]
    else:
        s = sum(weights)
        weights = [wi / s for wi in weights]

    ssim_vals = []
    m = mask
    for _ in range(max_levels):
        ssim_map = tf.image.ssim(
            yt, yp, max_val=1.0, filter_size=filter_size, k1=k1, k2=k2
        )
        ssim_vals.append(ssim_map)

        yt = _avg_pool_2x(yt)
        yp = _avg_pool_2x(yp)
        if m is not None:
            m = _avg_pool_2x(m)

    ssim_vals = [tf.clip_by_value(v, 0.0, 1.0) for v in ssim_vals]
    ms = 0.0
    for wi, vi in zip(weights, ssim_vals):
        ms = ms + wi * vi

    tf.debugging.assert_all_finite(ms, "ms-ssim-stable returned NaN/Inf")
    loss = 1.0 - tf.reduce_mean(ms)
    tf.debugging.assert_all_finite(loss, "MS-SSIM-stable produced NaN/Inf")
    return loss


def composite_loss(
    ssim_loss_weight=0.16,
    use_ms_ssim=True,
    grad_loss_weight=0.05,
    ssim_max_levels=5,
    ssim_filter_size=7,
    ssim_k1=0.02,
    ssim_k2=0.04,
    perceptual_weight=0.0,
    vgg_model=None,
):
    """MAE + (MS-)SSIM + gradient + optional perceptual loss."""

    def _loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        yt, yp = _to_float01(y_true, y_pred)
        mae = tf.reduce_mean(tf.abs(yt - yp))

        if use_ms_ssim:
            ssim_term = ssim_multiscale_stable(
                yt,
                yp,
                value_range=(0.0, 1.0),
                max_levels=ssim_max_levels,
                filter_size=ssim_filter_size,
                k1=ssim_k1,
                k2=ssim_k2,
            )
        else:
            ssim = tf.image.ssim(
                tf.clip_by_value(yt, 0.0, 1.0),
                tf.clip_by_value(yp, 0.0, 1.0),
                max_val=1.0,
                filter_size=7,
                k1=0.02,
                k2=0.04,
            )
            ssim_term = 1.0 - tf.reduce_mean(ssim)

        grad_term = gradient_l1_loss(yt, yp) if grad_loss_weight > 0 else 0.0
        total = mae + ssim_loss_weight * ssim_term + grad_loss_weight * grad_term

        if (perceptual_weight > 0.0) and (vgg_model is not None):
            p_loss = perceptual_loss(yt, yp, vgg_model, mask=None)
            total = total + perceptual_weight * p_loss

        tf.debugging.assert_all_finite(total, "Loss has NaN/Inf")
        return total

    return _loss
