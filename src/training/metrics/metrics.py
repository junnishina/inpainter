import tensorflow as tf


def weighted_ssim_metric(y_true4, y_pred3, max_val=1.0, filter_size=7):
    """
    y_true4: (..., 4) first 3ch = GT, 4th = mask (1: hole, 0: context)
    y_pred3: (..., 3) prediction (0-1)
    """
    y_true = y_true4[..., :3]
    mask = y_true4[..., 3:4]  # 1:hole

    y_pred = tf.clip_by_value(y_pred3, 0.0, 1.0)
    y_pred_hole_only = mask * y_pred + (1.0 - mask) * y_true

    ssim = tf.image.ssim(
        y_true, y_pred_hole_only, max_val=max_val, filter_size=filter_size
    )

    hole_area = tf.reduce_mean(mask, axis=[1, 2, 3]) + 1e-6
    ssim_weighted = tf.reduce_sum(ssim * hole_area) / tf.reduce_sum(hole_area)
    return ssim_weighted


class WeightedSSIMMetric(tf.keras.metrics.Metric):
    def __init__(self, name="weighted_ssim_metric", filter_size=7, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filter_size = int(filter_size)
        self.sum = self.add_weight(name="sum", initializer="zeros", dtype=tf.float32)
        self.den = self.add_weight(name="den", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true3, y_pred3, sample_weight=None):
        y_true = tf.clip_by_value(tf.cast(y_true3, tf.float32), 0.0, 1.0)
        y_pred = tf.clip_by_value(tf.cast(y_pred3, tf.float32), 0.0, 1.0)

        mask = None
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            if tf.rank(sw) == 3:
                sw = sw[..., tf.newaxis]
            if sw.shape[-1] > 1:
                sw = sw[..., :1]
            sw_min = tf.reduce_min(sw)
            sw_max = tf.reduce_max(sw)
            denom = tf.maximum(sw_max - sw_min, 1e-6)
            mask = tf.clip_by_value((sw - sw_min) / denom, 0.0, 1.0)

        if mask is not None:
            y_pred_hole_only = mask * y_pred + (1.0 - mask) * y_true
            ssim = tf.image.ssim(
                y_true, y_pred_hole_only, max_val=1.0, filter_size=self.filter_size
            )
            hole_area = tf.reduce_mean(mask, axis=[1, 2, 3]) + 1e-6
            self.sum.assign_add(tf.reduce_sum(ssim * hole_area))
            self.den.assign_add(tf.reduce_sum(hole_area))
        else:
            ssim = tf.image.ssim(
                y_true, y_pred, max_val=1.0, filter_size=self.filter_size
            )
            self.sum.assign_add(tf.reduce_sum(ssim))
            self.den.assign_add(tf.cast(tf.shape(ssim)[0], tf.float32))

    def result(self):
        return tf.where(self.den > 0.0, self.sum / self.den, 0.0)

    def reset_state(self):
        self.sum.assign(0.0)
        self.den.assign(0.0)

    def reset_states(self):
        self.reset_state()


def psnr_metric(y_true, y_pred):
    yt = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    yp = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    v = tf.image.psnr(yt, yp, max_val=1.0)
    v = tf.where(tf.math.is_inf(v), tf.fill(tf.shape(v), 99.0), v)
    return tf.reduce_mean(v)


def ssim_metric(y_true, y_pred):
    yt = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    yp = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    v = tf.image.ssim(yt, yp, max_val=1.0)
    return tf.reduce_mean(v)


def psnr_raw(y_true, y_pred):
    yt = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    yp = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(yt, yp, max_val=1.0))
