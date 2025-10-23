#!/usr/bin/env python3
import os
import glob
import math
import argparse
import random
import numpy as np

os.environ["SM_FRAMEWORK"] = (
    "tf.keras"  # segmentation_models のバックエンド指定（必須）
)

import tensorflow as tf
from tensorflow import keras
from segmentation_models import Unet

import albumentations as A
import cv2
from freeze import freeze


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _str2bool(x: str) -> bool:
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got: {x}")


def weights_arg(s: str):
    if s is None:
        return None
    s_norm = str(s).strip().lower()
    if s_norm in ("none", "", "null", "nil"):
        return None
    if s_norm == "imagenet":
        return "imagenet"
    raise argparse.ArgumentTypeError(f"Unsupported weights value: {s}")


# 追加: 環境変数から data_dir の既定を拾う
def get_default_data_dir():
    return os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")


# 改良版 list_images（拡張子の大文字小文字を吸収し、デバッグ出力付き）
def list_images(data_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data dir does not exist: {data_dir}")

    wanted = set(e.lower() for e in exts)
    files = []
    if os.path.isfile(data_dir):
        # 単一ファイル指定も許容
        if os.path.splitext(data_dir)[1].lower() in wanted:
            files = [data_dir]
    else:
        for root, _, fnames in os.walk(data_dir):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in wanted:
                    files.append(os.path.join(root, f))

    files = [f for f in files if os.path.isfile(f)]
    if not files:
        # デバッグ出力（上位2階層を覗く）
        print(f"[debug] No images found under: {data_dir}")
        try:
            lvl1 = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir)])
            print("[debug] level-1:", lvl1[:50])
            for d in lvl1[:10]:
                if os.path.isdir(d):
                    print(f"[debug] {d} -> {os.listdir(d)[:20]}")
        except Exception as e:
            print(f"[debug] listing error: {e}")
        raise FileNotFoundError("No images found")
    return sorted(files)


def build_albu_transforms(img_size, use_blur=False):
    # 幾何は source/target 共通
    shared_geom = A.Compose(
        [
            A.SmallestMaxSize(max_size=img_size, p=1.0),
            A.CenterCrop(
                height=img_size, width=img_size, p=1.0
            ),  # リサイズ起因の歪みを避けたい場合
            # あるいは：A.Resize(img_size, img_size)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
        ]
    )

    # 欠損（CoarseDropout）は ReplayCompose でパラメータを保存
    coarse_dropout = A.ReplayCompose(
        [
            A.CoarseDropout(
                max_holes=8,  # 欠損の個数
                min_holes=2,
                max_height=int(img_size * 0.25),
                max_width=int(img_size * 0.25),
                min_height=int(img_size * 0.06),
                min_width=int(img_size * 0.06),
                fill_value=0,  # 欠損は 0 埋め（後でマスク生成にも利用）
                p=1.0,
            )
        ],
        p=1.0,
    )

    # target のみフォトメトリック劣化
    photometric = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.08, contrast_limit=0.08, p=0.6
            ),
            A.GaussNoise(var_limit=(2.0, 10.0), p=0.5),
            A.MotionBlur(blur_limit=3, p=0.2) if use_blur else A.NoOp(),
        ]
    )

    return shared_geom, coarse_dropout, photometric


def read_image_cv2(path, img_size):
    # BGR -> RGB, float32 [0,1]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def make_pair_with_albu_np(path_bytes, img_size, use_blur):
    path = path_bytes.decode("utf-8")
    img = read_image_cv2(path, img_size)

    shared_geom, cd, photo = build_albu_transforms(img_size, use_blur=use_blur)

    # 幾何（共有）
    out = shared_geom(image=img)
    base = out["image"]

    # CoarseDropout（replay取得）
    cd_out = cd(image=base)
    tgt_cd = cd_out["image"]
    replay = cd_out["replay"]

    # マスク生成（同じ CoarseDropout を mask にだけ適用）
    mask_init = np.ones((base.shape[0], base.shape[1], 1), dtype=np.uint8) * 255
    mask_cd = A.ReplayCompose.replay(replay, image=mask_init)["image"]  # uint8 [0,255]
    mask = (mask_cd > 127).astype(np.float32)  # 1=有効, 0=欠損

    # フォトメトリックは target のみ（mask には適用しない）
    tgt = photo(image=tgt_cd)["image"]

    # 正規化 [0,1]
    src = base.astype(np.float32) / 255.0
    tgt = tgt.astype(np.float32) / 255.0

    return tgt, src, mask.astype(np.float32)


def tf_make_pair(path, img_size, use_blur):
    tgt, src, mask = tf.numpy_function(
        func=make_pair_with_albu_np,
        inp=[path, img_size, use_blur],
        Tout=[tf.float32, tf.float32, tf.float32],
    )
    # 形状を静的に与える
    img_size = (
        int(img_size.numpy()) if isinstance(img_size, tf.Tensor) else int(img_size)
    )
    tgt.set_shape([img_size, img_size, 3])
    src.set_shape([img_size, img_size, 3])
    mask.set_shape([img_size, img_size, 1])
    return {"tgt": tgt, "mask": mask}, src


def build_dataset(files, img_size, batch_size, shuffle=True, use_blur=False):
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(files), reshuffle_each_iteration=True)

    # py_function はスカラー以外の Python 値を渡しづらいので、テンソルで渡す
    img_size_t = tf.constant(img_size, dtype=tf.int64)
    use_blur_t = tf.constant(1 if use_blur else 0, dtype=tf.int64)

    def _map(path):
        return tf_make_pair(path, img_size_t, use_blur_t)

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_unet(img_size, backbone, weights):
    model = Unet(
        backbone_name=backbone,
        encoder_weights=weights,
        input_shape=(img_size, img_size, 3),
        classes=3,
        activation="sigmoid",  # 出力 [0,1]
        decoder_use_batchnorm=True,
    )
    return model


class PSNRMetric(keras.metrics.Metric):
    def __init__(self, name="psnr", max_val=1.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_val = tf.cast(max_val, tf.float32)
        self.sum_psnr = self.add_weight(name="sum_psnr", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
        mse = tf.maximum(mse, tf.keras.backend.epsilon())
        psnr = 10.0 * tf.math.log((self.max_val**2) / mse) / tf.math.log(10.0)
        if sample_weight is not None:
            psnr *= tf.cast(sample_weight, tf.float32)
        self.sum_psnr.assign_add(tf.reduce_sum(psnr))
        self.count.assign_add(tf.cast(tf.size(psnr), tf.float32))

    def result(self):
        return tf.where(self.count > 0, self.sum_psnr / self.count, 0.0)

    def reset_state(self):
        self.sum_psnr.assign(0.0)
        self.count.assign(0.0)


def _split_y_pack(ypack, y_channels=None):
    # ypack: (y_true, mask) または Tensor
    if isinstance(ypack, (tuple, list)):
        return ypack[0], ypack[1]
    # Tensor の場合はチャネルで分割
    if y_channels is not None:
        y_true = ypack[..., :y_channels]
        mask = ypack[..., y_channels : y_channels + 1]
    else:
        # 既定: 最後の1chをマスクとみなす
        y_true = ypack[..., :-1]
        mask = ypack[..., -1:]
    return y_true, mask


class InpaintModel(keras.Model):
    # 追加: target_channels を指定できるように（既定は自動推定＝最後1chをマスク）
    def __init__(self, unet, hole_weight=5.0, target_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.unet = unet
        self.hole_weight = float(hole_weight)
        self.target_channels = target_channels
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.psnr_metric = PSNRMetric(name="psnr")

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric, self.psnr_metric]


def _broadcast_mask(mask, like):
    # mask: [B,H,W,1 or C?], like: [B,H,W,C]
    mask = tf.cast(mask, tf.float32)
    if tf.shape(mask)[-1] == 1 and tf.shape(like)[-1] > 1:
        mask = tf.broadcast_to(mask, tf.shape(like))
    return mask

    def _make_input_from_target(y_true, mask, fill_value=0.0):
        # 欠損部を fill_value（既定0）で埋める
        mask = _broadcast_mask(mask, y_true)  # 1ch→Cchへ拡張
        inv = 1.0 - mask
        fill = tf.cast(fill_value, tf.float32)
        x = y_true * inv + fill * mask
        return x

    def train_step(self, data):
        # data が dict({'tgt': y, 'mask': m}) または (x, (y, m)) のどちらでも動く
        if isinstance(data, dict):
            y_true = data["tgt"]
            mask = data["mask"]
            x = _make_input_from_target(y_true, mask, fill_value=0.0)
        else:
            x, ypack = data
            if isinstance(ypack, dict):
                y_true = ypack["tgt"]
                mask = ypack["mask"]
            else:
                # ypack が (y, mask) か、結合Tensor（最後1chがmask）想定
                if isinstance(ypack, (tuple, list)):
                    y_true, mask = ypack
                else:
                    # ypack が結合Tensor [C_y + 1ch] のとき（最後の1chをmaskとみなす）
                    y_true, mask = ypack[..., :-1], ypack[..., -1:]
            x = _make_input_from_target(y_true, mask, fill_value=0.0)

        with tf.GradientTape() as tape:
            y_pred = tf.clip_by_value(self.unet(x, training=True), 0.0, 1.0)
            loss = self.compute_loss(y_true, y_pred, mask)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y_true, y_pred)
        self.psnr_metric.update_state(y_true, y_pred)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result(),
            "psnr": self.psnr_metric.result(),
        }

    def test_step(self, data):
        if isinstance(data, dict):
            y_true = data["tgt"]
            mask = data["mask"]
            x = _make_input_from_target(y_true, mask, fill_value=0.0)
        else:
            x, ypack = data
            if isinstance(ypack, dict):
                y_true = ypack["tgt"]
                mask = ypack["mask"]
            else:
                if isinstance(ypack, (tuple, list)):
                    y_true, mask = ypack
                else:
                    y_true, mask = ypack[..., :-1], ypack[..., -1:]
            x = _make_input_from_target(y_true, mask, fill_value=0.0)

        y_pred = tf.clip_by_value(self.unet(x, training=False), 0.0, 1.0)
        loss = self.compute_loss(y_true, y_pred, mask)

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y_true, y_pred)
        self.psnr_metric.update_state(y_true, y_pred)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result(),
            "psnr": self.psnr_metric.result(),
        }

    def compute_pixel_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3])  # L1

    def compute_loss(self, y_true, y_pred, mask):
        mask = tf.cast(mask, tf.float32)
        inv = 1.0 - mask
        hole = self.compute_pixel_loss(y_true * mask, y_pred * mask)
        valid = self.compute_pixel_loss(y_true * inv, y_pred * inv)
        hole_area = tf.reduce_mean(mask, axis=[1, 2, 3]) + tf.keras.backend.epsilon()
        valid_area = tf.reduce_mean(inv, axis=[1, 2, 3]) + tf.keras.backend.epsilon()
        hole = hole / hole_area
        valid = valid / valid_area
        return tf.reduce_mean(self.hole_weight * hole + valid)


def cosine_annealing_schedule(initial_lr, steps_per_epoch, epochs, alpha=0.0):
    total_steps = steps_per_epoch * epochs
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=alpha,  # 最終 lr = initial_lr * alpha
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=get_default_data_dir(),
        help="入力画像ディレクトリ（既定: SM_CHANNEL_TRAIN or /opt/ml/input/data/train）",
    )
    parser.add_argument("--output_dir", type=str, default="./ssl2_outputs")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--initial_lr", type=float, default=2e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hole_weight", type=float, default=5.0)
    parser.add_argument("--use_blur", type=int, default=0)
    parser.add_argument(
        "--weights",
        type=weights_arg,
        default="imagenet",
        help='encoder初期重み: "imagenet" または "none"（大小文字・空文字可）',
    )
    parser.add_argument("--backbone", type=str, default="efficientnetb1")
    parser.add_argument("--train_stage_count", type=int, default=1)
    parser.add_argument(
        "--freeze_bn", type=int, default=1, help="BatchNormも凍結するか（1/0）"
    )
    parser.add_argument(
        "--print_frozen_layers", type=int, default=0, help="凍結対象を表示（1/0）"
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

    args, unknown = parser.parse_known_args()  # 未知引数が注入されても落ちない
    if unknown:
        print(f"[warn] Ignored unknown args: {unknown}")
    print(f"[info] data_dir={args.data_dir}")
    print(f"[info] model_dir={args.model_dir}")

    set_seed(args.seed)
    # out_dir = args.model_dir or args.output_dir  # 明示されたら model_dir を優先
    out_dir = args.output_dir  # model_dir に勝手なs3 uriが注入されるので無視
    os.makedirs(out_dir, exist_ok=True)

    files = list_images(args.data_dir)
    random.Random(args.seed).shuffle(files)
    n_total = len(files)
    n_val = max(1, int(n_total * args.val_split))
    val_files = files[:n_val]
    train_files = files[n_val:]
    print(f"Found {n_total} images. Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = build_dataset(
        train_files,
        args.img_size,
        args.batch_size,
        shuffle=True,
        use_blur=bool(args.use_blur),
    )
    val_ds = build_dataset(
        val_files,
        args.img_size,
        args.batch_size,
        shuffle=False,
        use_blur=bool(args.use_blur),
    )

    base_unet = build_unet(args.img_size, args.backbone, args.weights)

    # ステージ凍結を適用（backbone='efficientnetb1'）
    freeze(
        model=base_unet,
        train_stage_count=args.train_stage_count,
        backbone="efficientnetb1",
        total_stages=7,  # EfficientNetB1 は7ステージ想定
        freeze_fpn=False,  # デコーダ/FPNは学習する
        freeze_bn=bool(args.freeze_bn),
        print_frozen_layers=bool(args.print_frozen_layers),
    )

    model = InpaintModel(base_unet, hole_weight=args.hole_weight)

    steps_per_epoch = math.ceil(len(train_files) / args.batch_size)
    lr_schedule = cosine_annealing_schedule(
        args.initial_lr, steps_per_epoch, args.epochs, alpha=0.0
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer)

    ckpt_path = os.path.join(out_dir, "best_ssl2_unet_efficientnetb1.weights.h5")
    csv_log = os.path.join(out_dir, "train_log.csv")
    final_path = os.path.join(out_dir, "ssl2_unet_efficientnetb1_final.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(csv_log),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ベスト重みをロードして最終保存
    model.unet.load_weights(ckpt_path)
    final_path = final_path
    model.unet.save_weights(final_path)
    print(f"Saved final weights to: {final_path}")


if __name__ == "__main__":
    main()
