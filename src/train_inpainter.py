import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import glob
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import cv2
import albumentations as A
import segmentation_models as sm

# 画像データ形式は明示的に NHWC
K.set_image_data_format("channels_last")

# Grappler のレイアウト最適化を無効化
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

# XLA を使っている場合は無効化（保険）
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

# mixed_precision.set_global_policy('mixed_float16')
mixed_precision.set_global_policy("float32")  # 既に 'mixed_float16' なら変更


def _to_uint8_rgb(img01: np.ndarray) -> np.ndarray:
    """[0,1]のRGB(float32) → uint8 RGB"""
    img = np.clip(img01 * 255.0, 0.0, 255.0).astype(np.uint8)
    return img


def save_preview_batch(
    x_raw01_b, y_b, y_pred_b, out_dir: str, prefix: str = "val", max_items: int = 16
):
    """
    x_raw01_b, y_b, y_pred_b: (B,H,W,3), [0,1] の想定
    [input | pred | target] を横連結して PNG 保存
    """
    os.makedirs(out_dir, exist_ok=True)
    b = min(x_raw01_b.shape[0], max_items)
    for i in range(b):
        in_img = _to_uint8_rgb(x_raw01_b[i])
        pr_img = _to_uint8_rgb(y_pred_b[i])
        gt_img = _to_uint8_rgb(y_b[i])

        concat = np.concatenate([in_img, pr_img, gt_img], axis=1)  # 横連結（W方向）
        # OpenCVはBGR期待なので変換して保存
        cv2.imwrite(
            os.path.join(out_dir, f"{prefix}_{i:03d}.png"),
            cv2.cvtColor(concat, cv2.COLOR_RGB2BGR),
        )


def weights_arg(s: str):
    if s is None:
        return None
    s_norm = str(s).strip().lower()
    if s_norm in ("none", "", "null", "nil"):
        return None
    if s_norm == "imagenet":
        return "imagenet"
    raise argparse.ArgumentTypeError(f"Unsupported weights value: {s}")


# ==== Losses: MAE + (SSIM or MS-SSIM) + Gradient (Sobel) ====

EPS = 1e-7


def _to_float01(y_true, y_pred):
    # AMP（mixed_precision）でもSSIM計算はfloat32で安定させる
    yt = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    yp = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return yt, yp


def ssim_loss(y_true, y_pred):
    yt, yp = _to_float01(y_true, y_pred)
    ssim = tf.image.ssim(yt, yp, max_val=1.0)
    return 1.0 - tf.reduce_mean(ssim)


def ms_ssim_loss(y_true, y_pred):
    yt, yp = _to_float01(y_true, y_pred)
    msssim = tf.image.ssim_multiscale(yt, yp, max_val=1.0)
    # まれに極端値でNaNが出る場合があるためnanを0に
    msssim = tf.where(tf.math.is_finite(msssim), msssim, tf.zeros_like(msssim))
    return 1.0 - tf.reduce_mean(msssim)


def gradient_l1_loss(y_true, y_pred):
    yt, yp = _to_float01(y_true, y_pred)
    sobel_true = tf.image.sobel_edges(yt)
    sobel_pred = tf.image.sobel_edges(yp)
    grad_true = tf.abs(sobel_true[..., 0]) + tf.abs(sobel_true[..., 1])
    grad_pred = tf.abs(sobel_pred[..., 0]) + tf.abs(sobel_pred[..., 1])
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))


def _avg_pool_2x(x):
    return tf.nn.avg_pool2d(x, ksize=2, strides=2, padding="SAME")


def ssim_multiscale_stable(
    y_true,
    y_pred,
    value_range=(0.0, 1.0),
    max_levels=3,
    filter_size=7,
    k1=0.05,
    k2=0.05,
    weights=None,  # Noneなら標準重みの先頭 levels
    mask=None,  # [N,H,W,1] 可。与えればマスク加重SSIM
):
    lo, hi = value_range
    yt = tf.cast(y_true, tf.float32)
    yp = tf.cast(y_pred, tf.float32)

    # [lo,hi] -> [0,1]
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
        # Python list 前提、合計1に正規化
        s = sum(weights)
        weights = [wi / s for wi in weights]

    ssim_vals = []
    m = mask
    for _ in range(max_levels):
        if m is not None:
            # マスク加重 SSIM
            # tf.image.ssim は全画素で平均するので、局所SSIMを直接重み付けはできない。
            # 代替として、マスクで画素を選別しつつ mean を調整（近似）。
            ssim_map = tf.image.ssim(
                yt, yp, max_val=1.0, filter_size=filter_size, k1=k1, k2=k2
            )
            # ssim_map: [N], すでに平均済なので厳密なマスク対応は困難。
            # → 近似として全体の ssim を使うか、マスクにより y を補間してから計算する方が実用的。
            ssim_vals.append(ssim_map)
        else:
            ssim_map = tf.image.ssim(
                yt, yp, max_val=1.0, filter_size=filter_size, k1=k1, k2=k2
            )
            ssim_vals.append(ssim_map)
        # 次スケールへ
        yt = _avg_pool_2x(yt)
        yp = _avg_pool_2x(yp)
        if m is not None:
            m = _avg_pool_2x(m)

    # 重み付き平均（安定のため clamp）
    ssim_vals = [tf.clip_by_value(v, 0.0, 1.0) for v in ssim_vals]
    ms = 0.0
    for wi, vi in zip(weights, ssim_vals):
        ms = ms + wi * vi

    tf.debugging.assert_all_finite(ms, "ms-ssim-stable returned NaN/Inf")
    loss = 1.0 - tf.reduce_mean(ms)
    tf.debugging.assert_all_finite(loss, "MS-SSIM-stable produced NaN/Inf")
    return loss


"""
def safe_ms_ssim(
    y_true, y_pred,
    value_range=(0.0, 1.0),   # (lo, hi) 入力の想定スケール
    clip=True,
    max_levels=5,
    filter_size=7,
    k1=0.02, k2=0.04
):
    lo, hi = value_range
    yt = tf.cast(y_true, tf.float32)
    yp = tf.cast(y_pred, tf.float32)

    # 必要なら入力スケールでのクリップ
    if clip:
        yt = tf.clip_by_value(yt, lo, hi)
        yp = tf.clip_by_value(yp, lo, hi)

    # SSIM用に [0,1] に正規化
    scale = tf.maximum(hi - lo, 1e-6)
    yt01 = (yt - lo) / scale
    yp01 = (yp - lo) / scale
    eps_margin = 1e-6
    yt01 = tf.clip_by_value(yt01, eps_margin, 1.0 - eps_margin)
    yp01 = tf.clip_by_value(yp01, eps_margin, 1.0 - eps_margin)

    # 入力サイズに応じて levels を調整（省略可、前述のロジックを流用）
    ms = tf.image.ssim_multiscale(
        yt01, yp01,
        max_val=1.0,
        filter_size=filter_size,
        filter_sigma=1.5,
        k1=k1, k2=k2
    )
    loss = 1.0 - tf.reduce_mean(ms)
    tf.debugging.assert_all_finite(loss, "MS-SSIM produced NaN/Inf")
    return loss
"""


def _infer_msssim_levels_from_static_shape(shape, filter_size=7, max_levels=5):
    # shape: TensorShape([N, H, W, C]) を想定
    h = shape[1]
    w = shape[2]
    if (h is None) or (w is None):
        # 動的形状の場合は保守的に小さめ
        return min(3, max_levels)

    min_hw = int(min(h, w))
    # 条件: min_hw >= filter_size * 2^(levels-1)
    # 最大 levels を探索
    levels = 1
    while levels < max_levels and (filter_size * (2**levels)) <= min_hw:
        levels += 1
    return max(1, min(levels, max_levels))


def safe_ms_ssim(
    y_true,
    y_pred,
    value_range=(0.0, 1.0),  # 出力/教師が [0,1] のとき (0,1)
    max_levels=5,
    filter_size=7,
    k1=0.02,
    k2=0.04,
):
    # 1) スケール合わせ（[0,1]）とクリップ（MS-SSIM は物理スケール前提）
    lo, hi = value_range
    yt = tf.cast(y_true, tf.float32)
    yp = tf.cast(y_pred, tf.float32)
    scale = tf.maximum(hi - lo, 1e-6)
    yt = (yt - lo) / scale
    yp = (yp - lo) / scale
    eps_margin = 1e-6
    yt = tf.clip_by_value(yt, eps_margin, 1.0 - eps_margin)
    yp = tf.clip_by_value(yp, eps_margin, 1.0 - eps_margin)

    # 2) levels は静的形状から推定（動的は 3 にフォールバック）
    levels = _infer_msssim_levels_from_static_shape(
        yt.shape, filter_size=filter_size, max_levels=max_levels
    )

    # 3) power_factors は Python リストで固定長に
    power_factors = [1.0 / levels] * levels

    def assert_finite(name, t):
        tf.debugging.assert_all_finite(t, f"{name} has NaN/Inf")

    # safe_ms_ssim 内、ssim_multiscale を呼ぶ直前に:
    assert_finite("yt_before_ssim", yt)
    assert_finite("yp_before_ssim", yp)

    # 4) MS-SSIM 計算
    """
    ms = tf.image.ssim_multiscale(
        yt, yp,
        max_val=1.0,
        power_factors=power_factors,      # ここは list が必須（len() が呼ばれる）
        filter_size=filter_size,
        filter_sigma=1.5,
        k1=k1, k2=k2
    )
    loss = 1.0 - tf.reduce_mean(ms)
    """
    ssim = tf.image.ssim(yt, yp, max_val=1.0, filter_size=7, k1=0.02, k2=0.04)
    loss = 1.0 - tf.reduce_mean(ssim)
    tf.debugging.assert_all_finite(loss, "MS-SSIM produced NaN/Inf")
    return loss


def composite_loss(
    ssim_loss_weight=0.16,
    use_ms_ssim=True,
    grad_loss_weight=0.05,
):
    def _loss(y_true, y_pred):
        yt, yp = _to_float01(y_true, y_pred)
        mae = tf.reduce_mean(tf.abs(yt - yp))

        # ssim_term = ms_ssim_loss(yt, yp) if use_ms_ssim else ssim_loss(yt, yp)
        if use_ms_ssim:
            # ssim_term = safe_ms_ssim(
            ssim_term = ssim_multiscale_stable(
                yt,
                yp,
                value_range=(0.0, 1.0),
                max_levels=5,  # 必要に応じて 3〜4 に
                filter_size=11,  # 7,  # 7 を推奨（小パッチ安定）
                k1=0.02,
                k2=0.04,
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
        # 最終的に非有限は抑止
        # return tf.where(tf.math.is_finite(total), total, tf.zeros_like(total))
        # [debug] ここで即チェック（fail-fast）
        tf.debugging.assert_all_finite(total, "Loss has NaN/Inf")
        return total

    return _loss


# メトリクス
def psnr_metric(y_true, y_pred):
    yt = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    yp = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    v = tf.image.psnr(yt, yp, max_val=1.0)
    # v = tf.where(tf.math.is_nan(v), tf.zeros_like(v), v)
    v = tf.where(tf.math.is_inf(v), tf.fill(tf.shape(v), 99.0), v)
    return tf.reduce_mean(v)


def ssim_metric(y_true, y_pred):
    yt = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    yp = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    v = tf.image.ssim(yt, yp, max_val=1.0)
    # v = tf.where(tf.math.is_finite(v), v, tf.zeros_like(v))
    return tf.reduce_mean(v)


def psnr_raw(y_true, y_pred):
    yt = tf.clip_by_value(tf.cast(y_true, tf.float32), 0.0, 1.0)
    yp = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(yt, yp, max_val=1.0))


def list_image_files(
    root_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    return files


def read_rgb_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR, 可能ならA含む
    if img is None:
        raise ValueError(f"cv2.imread failed: {path}")
    # グレースケール対応
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 4ch対応
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # BGR→RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 連続化（Albumentationsは連続配列が安全）
    return np.ascontiguousarray(img)


class RectDropout(A.ImageOnlyTransform):
    """
    ランダムな矩形穴を num_holes 個あける簡易欠損。
    画像サイズ比で穴サイズを指定するので解像度に依らず安定。
    Albumentationsバージョン非依存
    """

    def __init__(
        self,
        num_holes=4,
        min_h_ratio=0.05,
        max_h_ratio=0.20,
        min_w_ratio=0.05,
        max_w_ratio=0.20,
        fill_value=0,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.num_holes = int(num_holes)
        self.min_h_ratio = float(min_h_ratio)
        self.max_h_ratio = float(max_h_ratio)
        self.min_w_ratio = float(min_w_ratio)
        self.max_w_ratio = float(max_w_ratio)
        self.fill_value = fill_value

    def apply(self, img, **params):
        h, w = img.shape[:2]
        out = img.copy()
        rng = np.random.default_rng()
        for _ in range(self.num_holes):
            rh = rng.integers(
                max(1, int(h * self.min_h_ratio)), max(2, int(h * self.max_h_ratio) + 1)
            )
            rw = rng.integers(
                max(1, int(w * self.min_w_ratio)), max(2, int(w * self.max_w_ratio) + 1)
            )
            y1 = rng.integers(0, max(1, h - rh + 1))
            x1 = rng.integers(0, max(1, w - rw + 1))
            out[y1 : y1 + rh, x1 : x1 + rw] = self.fill_value
        return out

    def get_transform_init_args_names(self):
        return (
            "num_holes",
            "min_h_ratio",
            "max_h_ratio",
            "min_w_ratio",
            "max_w_ratio",
            "fill_value",
        )


class InpaintingDataset:
    """
    欠損復元用のデータセット:
      - 入力: CoarseDropout を適用した欠損画像（+共通の幾何Aug）
      - 教師: 欠損なし画像（同じ幾何Augのみ）
    """

    def __init__(
        self,
        files: List[str],
        image_size: Tuple[int, int] = (256, 256),
        backbone_name: str = "resnet34",
        shuffle: bool = True,
        coarse_dropout_params: dict = None,
        aug_prob: float = 0.8,
        seed: int = 42,
    ):
        self.files = files
        self.h, self.w = image_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        # 共有の幾何・色調Aug（教師にも適用）
        self.shared_aug = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.1),
                A.Affine(  # ShiftScaleRotate代替
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-10, 10),
                    shear=0,
                    interpolation=cv2.INTER_LINEAR,
                    # mode=cv2.BORDER_CONSTANT,  # 黒で埋める
                    # cval=(0, 0, 0),            # 黒（グレースケールなら 0）
                    fit_output=False,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.ColorJitter(p=0.05),
                A.Resize(height=self.h, width=self.w, interpolation=cv2.INTER_AREA),
            ]
        )

        # 欠損（入力のみに適用）
        if coarse_dropout_params is None:
            coarse_dropout_params = dict(
                num_holes=4,
                min_h_ratio=0.05,
                max_h_ratio=0.20,
                min_w_ratio=0.05,
                max_w_ratio=0.20,
                fill_value=0,  # ← masked_mae の fill_value と合わせる
                p=1.0,
            )
        self.dropout_aug = RectDropout(**coarse_dropout_params)
        """
        # debug
        self.shared_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(height=self.h, width=self.w, interpolation=cv2.INTER_AREA),
        ], p=1.0)
        
        # debug 欠損を一旦オフ
        self.dropout_aug = None
        """

        # 入力の前処理（エンコーダに一致）
        self.preprocess_input = sm.get_preprocessing(backbone_name)
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.files)

    def _load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _make_pair(self, path):
        x = read_rgb_image(path)  # uint8, HxWx3
        y = x.copy()  # 教師
        # まずは shared_aug のみ（uint8のまま適用）
        if self.shared_aug is not None:
            aug = self.shared_aug(image=x, mask=None)
            x = aug["image"]
            y = aug["image"]  # 同期させたいなら image を共用（マスクなし構成ならOK）

        # 欠損 RectDropout 適用
        if self.dropout_aug is not None:
            x = self.dropout_aug(image=x)["image"]

        # ここで欠損マスクを作成（fill_value=0 前提。違う場合は一致判定を調整）
        mask = (x[..., 0] == 0) & (x[..., 1] == 0) & (x[..., 2] == 0)  # True=穴
        mask = mask.astype(np.float32)[..., None]  # (H,W,1)

        # スケーリング（0-1正規化や sm の mean/std）を適用
        x_raw01 = x.astype(np.float32) / 255.0
        y01 = y.astype(np.float32) / 255.0
        x_pp = self.preprocess_input(
            x.astype(np.float32)
        )  # sm.get_preprocessing(backbone) に渡す入力は「0〜255スケールの float（dtype=float32）」が正解

        # 返却: x_pp（モデル入力）, y01（教師）, mask（sample_weight 用）
        return x_pp, y01, x_raw01, mask

    def generator(self):
        files = self.files[:]
        while True:
            if self.shuffle:
                self.rng.shuffle(files)
            produced = 0
            first_err = None
            for p in files:
                try:
                    x_pp, y, x_raw01, mask = self._make_pair(p)
                    produced += 1
                    yield x_pp, y, x_raw01, mask
                except Exception as e:
                    if first_err is None:
                        first_err = (p, repr(e))
                    continue
            if produced == 0:
                msg = "No valid samples produced in an epoch."
                if first_err:
                    msg += f" First error at: {first_err[0]} ; err={first_err[1]}"
                raise RuntimeError(msg)


def create_lr_schedule(
    initial_lr: float,
    steps_per_epoch: int,
    total_epochs: int,
    schedule_type: str = "cosine",
    final_lr_fraction: float = 0.01,
    t_max_epochs: float = 10.0,
    t_mul: float = 2.0,
    m_mul: float = 1.0,
    min_lr_fraction: float = 0.01,
):
    """
    schedule_type:
      - "cosine": tf.keras.optimizers.schedules.CosineDecay
      - "cosine_restarts": tf.keras.optimizers.schedules.CosineDecayRestarts
    """
    if schedule_type == "cosine":
        decay_steps = max(1, int(steps_per_epoch * total_epochs))
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=final_lr_fraction,
            name="cosine_decay",
        )
    elif schedule_type == "cosine_restarts":
        first_decay_steps = max(1, int(steps_per_epoch * t_max_epochs))
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=min_lr_fraction,  # サイクル最終LR = サイクル初期LR * alpha
            name="cosine_decay_restarts",
        )
    else:
        raise ValueError(f"Unsupported lr_schedule: {schedule_type}")


def build_model(
    image_size=(256, 256),
    backbone_name="resnet34",
    encoder_weights="imagenet",
    learning_rate=1e-4,  # float でも schedule でもOK
    ssim_loss_weight=0.16,
    use_ms_ssim=True,
    grad_loss_weight=0.05,
):
    model = sm.Unet(
        backbone_name=backbone_name,
        encoder_weights=encoder_weights,
        classes=3,
        activation="sigmoid",
        input_shape=(image_size[0], image_size[1], 3),
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        epsilon=1e-4,  # 1e-7 → 1e-4 に引き上げ（数値安定）
        clipnorm=1.0,
    )  # または clipvalue=1.0)

    # debug
    """
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[psnr_metric, ssim_metric]  # ここに NaN→0 の置換があるなら一旦外す/別名で生の値に
    )
    """
    # debug2
    model.compile(
        optimizer=optimizer,
        loss=composite_loss(
            ssim_loss_weight=0.05,
            use_ms_ssim=True,
            grad_loss_weight=0.02,
        ),
        metrics=[psnr_metric, ssim_metric, psnr_raw],
        # weighted_metrics はデバッグ中は外すと見通しが良い
    )
    """
    model.compile(
        optimizer=optimizer,
        loss=composite_loss(
            ssim_loss_weight=ssim_loss_weight,
            use_ms_ssim=use_ms_ssim,
            grad_loss_weight=grad_loss_weight,
        ),
        metrics=[psnr_metric,
                 ssim_metric,
                 psnr_raw], # 一時的に追加
        weighted_metrics=[psnr_metric, ssim_metric],
    )
    """

    return model


def make_tf_dataset(
    dataset: InpaintingDataset,
    batch_size: int,
    return_raw: bool = False,
    return_mask_weight: bool = False,
    hole_weight: float = 3.0,
    context_weight: float = 1.0,
    mask_fill_value: float = 0.0,
    mask_thr: float = 1.0 / 255.0,
):
    """
    - 通常学習/評価: (x_pp, y)
    - プレビュー: return_raw=True → (x_pp, y, x_raw01)
    - 欠損重み学習: return_mask_weight=True → (x_pp, y, sample_weight)
      sample_weight は (H,W,3) のピクセル重み（穴=hole_weight、その他=context_weight）

    制約:
    - return_raw と return_mask_weight は同時に True にできません
    - x_raw01 は [0,1] の float を想定（マスク判定のしきい値は mask_thr）
    """
    if return_raw and return_mask_weight:
        raise ValueError(
            "return_raw と return_mask_weight は同時に True にはできません。"
        )

    H, W, C = dataset.h, dataset.w, 3

    # 出力シグネチャ定義（分岐ごとに固定）
    if return_raw:
        output_signature = (
            tf.TensorSpec(shape=(H, W, C), dtype=tf.float32),  # x_pp
            tf.TensorSpec(shape=(H, W, C), dtype=tf.float32),  # y
            tf.TensorSpec(
                shape=(H, W, C), dtype=tf.float32
            ),  # x_raw01（可視化用, 3ch化）
        )
    elif return_mask_weight:
        output_signature = (
            tf.TensorSpec(shape=(H, W, C), dtype=tf.float32),  # x_pp
            tf.TensorSpec(shape=(H, W, C), dtype=tf.float32),  # y
            tf.TensorSpec(
                shape=(H, W, C), dtype=tf.float32
            ),  # sample_weight（y と同形状に統一）
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(H, W, C), dtype=tf.float32),  # x_pp
            tf.TensorSpec(shape=(H, W, C), dtype=tf.float32),  # y
        )

    def _make_hole_mask(x_raw01, fill_value, thr, use_all_channels=True):
        """
        x_raw01: [0,1] float, shape (H,W), (H,W,1) もしくは (H,W,C>=3)
        fill_value <= 0.5: 黒塗り（穴は小さい側）
        fill_value > 0.5 : 白塗り（穴は大きい側）
        返り値: (H,W,1) float32, {0,1}
        """
        arr = np.asarray(x_raw01, dtype=np.float32)
        # 2D → 3D
        if arr.ndim == 2:
            arr = arr[..., None]
        # 値域を [0,1] にクリップ
        arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if fill_value <= 0.5:
            cond = arr <= thr
        else:
            cond = arr >= (1.0 - thr)

        reduce_fn = np.all if use_all_channels else np.any
        hole = reduce_fn(cond, axis=-1, keepdims=True).astype(np.float32)  # (H,W,1)
        return hole

    def gen():
        for x_pp, y, x_raw01, hole_mask in dataset.generator():
            # クリップ＆型統一（安全策）
            # x_pp = np.clip(x_pp, 0.0, 1.0).astype(np.float32) # backbone正規化後のclipは排除
            x_pp = x_pp.astype(np.float32)
            y = np.clip(y, 0.0, 1.0).astype(np.float32)

            if return_raw:
                xr = x_raw01
                if xr.ndim == 2:
                    xr = xr[..., None]
                if xr.shape[-1] == 1:
                    xr = np.repeat(xr, 3, axis=-1)
                xr = np.clip(xr, 0.0, 1.0).astype(np.float32)
                yield x_pp, y, xr
            elif return_mask_weight:
                """
                hole_mask = _make_hole_mask(
                    x_raw01, fill_value=mask_fill_value, thr=mask_thr, use_all_channels=True
                )  # (H,W,1)

                # 重みマップ（穴:hole_weight, それ以外:context_weight）
                sw = hole_mask * float(hole_weight) + (1.0 - hole_mask) * float(context_weight)  # (H,W,1)

                # y と同形状（(H,W,3)）に拡張
                sw3 = np.repeat(sw, 3, axis=-1)

                # 数値安定化 + 全ゼロ防止
                eps = 1e-6
                sw3 = np.nan_to_num(sw3, nan=float(context_weight), posinf=float(hole_weight), neginf=0.0)
                sw3 += eps
                # debug
                if np.random.rand() < 0.01:
                    print(
                        f"[dbg sw] sum={float(sw3.sum()):.6g}, "
                        f"min={float(sw3.min()):.6g}, max={float(sw3.max()):.6g}, "
                        f"hole_px={int(hole_mask.sum())}"
                    )
                yield x_pp, y, sw3.astype(np.float32)
                """
                # dataset.generator() の第4戻り値は穴マスク (H,W,1)
                hm = np.clip(hole_mask.astype(np.float32), 0.0, 1.0)  # (H,W,1)
                sw = hm * float(hole_weight) + (1.0 - hm) * float(context_weight)
                sw3 = np.repeat(sw, 3, axis=-1).astype(np.float32)
                sw3 += 1e-6  # 数値安定化
                yield x_pp, y, sw3
            else:
                yield x_pp, y

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def masked_mae_np(
    y_true, y_pred, x_input_raw01, fill_value=0.0, thr=1 / 255.0, use_all_channels=True
):
    """
    安定版 masked MAE（numpy）
    - y_true, y_pred, x_input_raw01: np.ndarray, shape (B,H,W,C), 値域 [0,1] 想定
    - fill_value <= 0.5: 黒埋め判定（≤ thr）
      fill_value > 0.5: 白埋め判定（≥ 1 - thr）
    - use_all_channels=True: 穴判定は「全チャネルが条件を満たす」(AND) → 通常の黒/白塗りに一致
      False: 「いずれかのチャネルが条件を満たす」(OR)
    返り値: float（穴領域におけるピクセル平均MAE）
    """

    def _clip01(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(a, 0.0, 1.0)

    yt = _clip01(y_true)
    yp = _clip01(y_pred)
    xin = _clip01(x_input_raw01)

    # 形状チェック
    if yt.shape != yp.shape:
        raise ValueError(f"y_true.shape {yt.shape} != y_pred.shape {yp.shape}")
    if xin.ndim != 4:
        raise ValueError(f"x_input_raw01 must be 4D (B,H,W,C), got {xin.shape}")

    # マスク生成（C=1 でも C=3 でも動く）
    if fill_value <= 0.5:
        cond = xin <= thr
    else:
        cond = xin >= (1.0 - thr)

    if use_all_channels:
        # 全チャネル一致で穴とみなす（AND）
        hole = np.all(cond, axis=-1, keepdims=True).astype(np.float32)  # (B,H,W,1)
    else:
        # いずれかのチャネル一致で穴（OR）
        hole = np.any(cond, axis=-1, keepdims=True).astype(np.float32)  # (B,H,W,1)

    # ピクセル平均のL1誤差（チャネル平均）
    per_pixel_l1 = np.mean(np.abs(yt - yp), axis=-1, keepdims=True)  # (B,H,W,1)

    num = float(np.sum(hole))  # 穴ピクセル数
    if num < 1e-6:
        return 0.0  # 穴が無い場合は 0.0 を返す（NaN回避）

    val = float(np.sum(per_pixel_l1 * hole) / num)
    return val


class MaskedMAELogger(tf.keras.callbacks.Callback):
    def __init__(self, raw_ds, name="val", fill_value=0.0, thr=1 / 255.0):
        super().__init__()
        self.raw_iter = iter(raw_ds)
        self.name = name
        self.fill_value = fill_value
        self.thr = thr

    def on_epoch_end(self, epoch, logs=None):
        try:
            x_pp_b, y_b, x_raw01_b = next(self.raw_iter)
        except StopIteration:
            return
        y_pred_b = self.model.predict(x_pp_b, verbose=0)
        val = masked_mae_np(
            y_b.numpy(),
            y_pred_b,
            x_raw01_b.numpy(),
            fill_value=self.fill_value,
            thr=self.thr,
        )
        print(f"[eval] epoch {epoch+1}: masked MAE ({self.name}, 1 batch) = {val:.5f}")


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker system args
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--backbone-name", type=str, default="resnet34")
    parser.add_argument(
        "--encoder-weights",
        type=weights_arg,
        default="imagenet",
        help='encoder初期重み: "imagenet" または "none"（大小文字・空文字可）',
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "cosine_restarts"],
        help="Learning rate schedule type",
    )
    parser.add_argument(
        "--final_lr_fraction",
        type=float,
        default=0.01,
        help="[cosine] 最終LR = 初期LR * final_lr_fraction",
    )
    parser.add_argument(
        "--t_max_epochs",
        type=float,
        default=10.0,
        help="[cosine_restarts] 最初のサイクル長（エポック数）",
    )
    parser.add_argument(
        "--t_mul",
        type=float,
        default=2.0,
        help="[cosine_restarts] サイクル長の倍率（>1で周期が伸びる）",
    )
    parser.add_argument(
        "--m_mul",
        type=float,
        default=1.0,
        help="[cosine_restarts] リスタート毎の初期LR倍率（<1で段階的に下げる）",
    )
    parser.add_argument(
        "--min_lr_fraction",
        type=float,
        default=0.01,
        help="[cosine_restarts] 各サイクルの最終LR = サイクル初期LR * min_lr_fraction",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=0)  # 0→自動計算
    parser.add_argument("--val-steps", type=int, default=0)
    parser.add_argument("--aug-prob", type=float, default=0.9)
    parser.add_argument("--hole_weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)

    # SageMaker input/output channels
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--val",
        type=str,
        default=os.environ.get("SM_CHANNEL_VAL", "/opt/ml/input/data/val"),
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="/opt/ml/model",
        help="モデル成果物の保存先（SageMaker既定）",
    )
    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        type=str,
        default="/opt/ml/model",
        help="同上（ハイフン表記互換）",
    )
    parser.add_argument("--ssim_loss_weight", type=float, default=0.16)

    args, unknown = parser.parse_known_args()  # 未知引数が注入されても落ちない
    if unknown:
        print(f"[warn] Ignored unknown args: {unknown}")
    print(f"[info] model_dir={args.model_dir}")

    # out_dir = args.model_dir or args.output_dir  # 明示されたら model_dir を優先
    # out_dir = args.output_dir  # model_dir に勝手なs3 uriが注入されるので無視
    # os.makedirs(out_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    tf.get_logger().setLevel("INFO")
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    out_dir = sm_model_dir  # ローカル固定
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] using local model dir: {out_dir}")

    # データ収集
    train_files = list_image_files(args.train)
    if len(train_files) == 0:
        raise RuntimeError(f"No training images found under: {args.train}")
    val_files = list_image_files(args.val) if os.path.isdir(args.val) else []

    image_size = (args.image_size, args.image_size)

    # データセット
    train_ds_builder = InpaintingDataset(
        files=train_files,
        image_size=image_size,
        backbone_name=args.backbone_name,
        shuffle=True,
        aug_prob=args.aug_prob,
        seed=args.seed,
    )
    # 学習用（従来どおり2タプル）
    # train_ds = make_tf_dataset(train_ds_builder, args.batch_size, return_raw=False)
    # 欠損部を強調（穴=3.0, 背景=1.0）。白塗りマスクなら mask_fill_value=1.0 に変更。
    train_ds = make_tf_dataset(
        train_ds_builder,
        batch_size=args.batch_size,
        return_mask_weight=True,
        hole_weight=args.hole_weight,
        context_weight=1.0,
        mask_fill_value=0.0,
        mask_thr=1.0 / 255.0,
    )
    # masked_mae 用（3タプル）
    train_raw_ds = make_tf_dataset(train_ds_builder, args.batch_size, return_raw=True)

    val_ds_builder = InpaintingDataset(
        files=val_files,
        image_size=image_size,
        backbone_name=args.backbone_name,
        shuffle=False,
        aug_prob=args.aug_prob,
        seed=args.seed + 1,
    )
    # val_ds = make_tf_dataset(val_ds_builder, args.batch_size, return_raw=False) if len(val_files) > 0 else None
    val_ds = make_tf_dataset(
        val_ds_builder,
        batch_size=args.batch_size,
        return_mask_weight=False,  # もし検証でも重みを使いたいなら、context_weight>0 かつ eps 付与を必ず適用
        hole_weight=args.hole_weight,
        context_weight=1.0,
        mask_fill_value=0.0,
        mask_thr=1.0 / 255.0,
    )

    val_raw_ds = (
        make_tf_dataset(val_ds_builder, args.batch_size, return_raw=True)
        if len(val_files) > 0
        else None
    )

    # ステップ数自動計算（大きすぎると永遠に回るので上限を設定）
    """
    steps_per_epoch = args.steps_per_epoch or max(1, min(len(train_files) // args.batch_size, 5000))
    val_steps = None
    if val_ds is not None:
        val_steps = args.val_steps or max(1, min(len(val_files) // args.batch_size, 1000))
    """

    def available_batches(num_files, batch_size):
        # drop_remainder=False なので端数も1バッチとして数える
        return max(1, (num_files + batch_size - 1) // batch_size)

    train_batches = available_batches(len(train_files), args.batch_size)
    val_batches = (
        available_batches(len(val_files), args.batch_size) if len(val_files) > 0 else 0
    )

    steps_per_epoch = args.steps_per_epoch or min(train_batches, 5000)
    val_steps = (args.val_steps or min(val_batches, 1000)) if val_batches > 0 else None

    # 学習率スケジューラを生成
    lr_schedule = create_lr_schedule(
        initial_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        total_epochs=args.epochs,
        schedule_type=args.lr_schedule,
        final_lr_fraction=args.final_lr_fraction,  # cosine 用
        t_max_epochs=args.t_max_epochs,  # restarts 用
        t_mul=args.t_mul,  # restarts 用
        m_mul=args.m_mul,  # restarts 用
        min_lr_fraction=args.min_lr_fraction,  # restarts 用
    )

    # モデル生成（learning_rate に schedule を渡す）
    model = build_model(
        image_size=image_size,
        backbone_name=args.backbone_name,
        encoder_weights=(
            args.encoder_weights if args.encoder_weights != "None" else None
        ),
        learning_rate=lr_schedule,
        ssim_loss_weight=args.ssim_loss_weight,
    )
    ## debug
    train_it = iter(train_ds)
    xb = (
        next(train_it)[0]
        if isinstance(next(iter(train_ds)), (list, tuple))
        else next(train_it)
    )
    yp0 = model.predict(xb, verbose=0)
    print("pre-train y_pred has NaN?", np.isnan(yp0).any())

    # データ1バッチを取り出してモデルに通す
    tmp_ds = make_tf_dataset(train_ds_builder, args.batch_size, return_raw=False)
    x_b, y_b = next(iter(tmp_ds.take(1)))
    _ = model.predict(x_b, verbose=0)  # ここが通れば少なくとも1バッチはOK
    print(f"[sanity] got batch: x={x_b.shape}, y={y_b.shape}")

    # コールバック
    ckpt_path = os.path.join(out_dir, "model.h5")

    class LrLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            opt = self.model.optimizer
            lr_obj = opt.learning_rate

            current_lr = None
            try:
                # Keras Optimizer が内部で現在の減衰後LRを持っていればそれを使う
                if hasattr(opt, "_decayed_lr"):
                    current_lr = tf.keras.backend.get_value(opt._decayed_lr(tf.float32))
                else:
                    # schedule / callable なら step(=iterations) を渡して評価
                    if isinstance(
                        lr_obj, tf.keras.optimizers.schedules.LearningRateSchedule
                    ) or callable(lr_obj):
                        step = tf.cast(opt.iterations, tf.float32)
                        current_lr = tf.keras.backend.get_value(lr_obj(step))
                    else:
                        # ただの変数（ResourceVariableなど）の場合はそのまま値を読む
                        current_lr = tf.keras.backend.get_value(lr_obj)
            except Exception:
                # フォールバック: どうしても取れない場合にテンソル化して数値化
                current_lr = float(tf.convert_to_tensor(lr_obj).numpy())

            print(f"[lr] epoch {epoch+1}: {current_lr:.8e}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_ssim_metric" if val_ds is not None else "psnr_metric",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ssim_metric" if val_ds is not None else "psnr_metric",
            patience=8,
            mode="max",
            restore_best_weights=True,
        ),
        LrLogger(),
        tf.keras.callbacks.TerminateOnNaN(),  # [debug]
    ]
    if val_raw_ds is not None:
        callbacks.append(MaskedMAELogger(val_raw_ds, name="val", fill_value=0.0))

    # 学習

    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    """
    # debug
    history = model.fit(
        train_ds, 
        validation_data=val_ds,   # いまは sample_weight なしでOK
        epochs=1
    )
    """
    ## debug
    # 1) 検証データ1バッチの中身確認
    val_it = iter(val_ds)  # model.fit に渡しているのと同じ val_ds
    batch = next(val_it)
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x_pp, y, sw = batch
        print("x_pp", x_pp.dtype, x_pp.shape, np.nanmin(x_pp), np.nanmax(x_pp))
        print("y   ", y.dtype, y.shape, np.nanmin(y), np.nanmax(y))
        print("sw  ", sw.dtype, sw.shape, np.sum(sw), np.min(sw), np.max(sw))
        print(
            "sw sum/min/max:", float(np.sum(sw)), float(np.min(sw)), float(np.max(sw))
        )
    else:
        x_pp, y = batch
        print("x_pp", x_pp.dtype, x_pp.shape, np.nanmin(x_pp), np.nanmax(x_pp))
        print("y   ", y.dtype, y.shape, np.nanmin(y), np.nanmax(y))
        print("val_ds has no sample_weight")

    # 2) モデル出力の NaN 有無確認（検証の最初の1バッチ）
    y_pred = model.predict(x_pp, verbose=0)
    print(
        "y_pred",
        y_pred.dtype,
        y_pred.shape,
        np.isnan(y_pred).any(),
        np.nanmin(y_pred),
        np.nanmax(y_pred),
    )

    # 3) Keras 経由で1バッチだけ評価（sample_weightの影響も見る）
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        res = model.test_on_batch(x_pp, y, sample_weight=sw, return_dict=True)
    else:
        res = model.test_on_batch(x_pp, y, return_dict=True)
    print(res)
    ## debug end

    if val_raw_ds is not None:
        x_pp_b, y_b, x_raw01_b = next(iter(val_raw_ds))  # 1バッチ取り出し
        y_pred_b = model.predict(x_pp_b, verbose=0)  # 予測（[0,1]想定）
        mmae = masked_mae_np(
            y_b.numpy(), y_pred_b, x_raw01_b.numpy(), fill_value=0.0, thr=1 / 255.0
        )
        print(f"[eval] masked MAE (val, 1 batch): {mmae:.5f}")
    else:
        # val が無ければ train で代用
        x_pp_b, y_b, x_raw01_b = next(iter(train_raw_ds))
        y_pred_b = model.predict(x_pp_b, verbose=0)
        mmae = masked_mae_np(
            y_b.numpy(), y_pred_b, x_raw01_b.numpy(), fill_value=0.0, thr=1 / 255.0
        )
        print(f"[eval] masked MAE (train, 1 batch): {mmae:.5f}")

    # SavedModel と重み保存（SageMaker は /opt/ml/model 配下を成果物として取得）
    # model.save(os.path.join(out_dir, "saved_model"))

    # 任意: 最終エポックの h5 も保存
    # model.save(os.path.join(out_dir, "final.h5"))

    # ==== Preview generation ====
    try:
        preview_dir = os.path.join(out_dir, "preview")
        # 対象は val 優先、なければ train
        if val_files:
            preview_builder = InpaintingDataset(
                files=val_files,
                image_size=image_size,
                backbone_name=args.backbone_name,
                shuffle=False,
                aug_prob=args.aug_prob,
                seed=args.seed + 123,  # 乱数を固定して再現性確保
            )
            prefix = "val"
        else:
            preview_builder = InpaintingDataset(
                files=train_files,
                image_size=image_size,
                backbone_name=args.backbone_name,
                shuffle=False,
                aug_prob=args.aug_prob,
                seed=args.seed + 123,
            )
            prefix = "train"

        # Aug後かつ前処理前の入力（x_raw01）も得るため return_raw=True
        preview_ds = make_tf_dataset(preview_builder, args.batch_size, return_raw=True)
        x_pp_b, y_b, x_raw01_b = next(iter(preview_ds.take(1)))

        # ベストモデルを読み直して推論（compile不要）
        best_model_path = os.path.join(out_dir, "model.h5")
        best_model = tf.keras.models.load_model(best_model_path, compile=False)
        y_pred_b = best_model.predict(x_pp_b, verbose=0)  # 出力は [0,1]（sigmoid）

        # 保存
        save_preview_batch(
            x_raw01_b.numpy(),
            y_b.numpy(),
            y_pred_b,
            out_dir=preview_dir,
            prefix=prefix,
            max_items=16,
        )
        print(f"[preview] saved to: {preview_dir}")
    except Exception as e:
        print(f"[warn] preview generation skipped due to error: {repr(e)}")


if __name__ == "__main__":
    main()
