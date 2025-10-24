import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import glob
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import cv2
import albumentations as A
import segmentation_models as sm

from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

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


# 簡易メトリクス
def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)


# ==== Composite loss (MAE + SSIM) ====
def ssim_loss(y_true, y_pred):
    # y_true, y_pred ∈ [0,1]
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - tf.reduce_mean(ssim)


def composite_loss(ssim_loss_weight=0.16):
    def _loss(y_true, y_pred):
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return mae + ssim_loss_weight * ssim_loss(y_true, y_pred)

    return _loss


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
                # A.RandomBrightnessContrast(p=0.2),
                A.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.05,
                    hue=0.01,
                    p=0.8,
                ),
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
        self.shared_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(height=self.h, width=self.w, interpolation=cv2.INTER_AREA),
        ], p=1.0)
        
        # 欠損も一旦オフ（後で戻す）
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

        # 欠損
        if self.dropout_aug is not None:
            x = self.dropout_aug(image=x)["image"]

        # スケーリングは最後にまとめて
        x_raw01 = x.astype(np.float32) / 255.0
        y01 = y.astype(np.float32) / 255.0
        x_pp = self.preprocess_input(
            x.astype(np.float32)
        )  # sm.get_preprocessing に渡すのは float でOK
        return x_pp, y01, x_raw01

    def generator(self):
        files = self.files[:]
        while True:
            if self.shuffle:
                self.rng.shuffle(files)
            produced = 0
            first_err = None
            for p in files:
                try:
                    x_pp, y, x_raw01 = self._make_pair(p)
                    produced += 1
                    yield x_pp, y, x_raw01
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
):
    model = sm.Unet(
        backbone_name=backbone_name,
        encoder_weights=encoder_weights,
        classes=3,
        activation="sigmoid",
        input_shape=(image_size[0], image_size[1], 3),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=composite_loss(ssim_loss_weight=ssim_loss_weight),
        metrics=[psnr_metric, ssim_metric],
    )
    return model


def make_tf_dataset(
    dataset: InpaintingDataset, batch_size: int, return_raw: bool = False
):
    if return_raw:
        output_signature = (
            tf.TensorSpec(shape=(dataset.h, dataset.w, 3), dtype=tf.float32),  # x_pp
            tf.TensorSpec(shape=(dataset.h, dataset.w, 3), dtype=tf.float32),  # y
            tf.TensorSpec(shape=(dataset.h, dataset.w, 3), dtype=tf.float32),  # x_raw01
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(dataset.h, dataset.w, 3), dtype=tf.float32),  # x_pp
            tf.TensorSpec(shape=(dataset.h, dataset.w, 3), dtype=tf.float32),  # y
        )

    def gen():
        for x_pp, y, x_raw01 in dataset.generator():
            if return_raw:
                yield x_pp, y, x_raw01
            else:
                yield x_pp, y

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    # ds = ds.batch(batch_size, drop_remainder=False)  # ← False に
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # ds = ds.prefetch(1)  # ← AUTOTUNE ではなく 1 に（デバッグ安定）
    return ds


def masked_mae_np(y_true, y_pred, x_input_raw01, fill_value=0.0, thr=1 / 255.0):
    """
    y_true, y_pred, x_input_raw01: numpy arrays, shape (B,H,W,3), [0,1]
    欠損マスクは x_input_raw01 が fill_value 近傍の画素を穴とみなす
    """
    import numpy as np

    mask = (np.abs(x_input_raw01 - fill_value) <= thr).astype(np.float32)
    mask = (np.max(mask, axis=-1, keepdims=True) > 0.5).astype(
        np.float32
    )  # any channel
    num = np.sum(mask) + 1e-8
    return float(np.sum(np.abs(y_true - y_pred) * mask) / num)


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
    train_ds = make_tf_dataset(train_ds_builder, args.batch_size, return_raw=False)
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
    val_ds = (
        make_tf_dataset(val_ds_builder, args.batch_size, return_raw=False)
        if len(val_files) > 0
        else None
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
            patience=5,
            mode="max",
            restore_best_weights=True,
        ),
        LrLogger(),
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
