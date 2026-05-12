# %%
import os

os.environ["KERAS_BACKEND"] = "torch"
# Prevent OpenCV from attempting to load a Qt/xcb GUI plugin in a
# headless / multi-threaded environment (DataLoader workers).
os.environ.setdefault("DISPLAY", ":0")
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib

matplotlib.use("Agg")  # non-interactive, file-only backend – must be before pyplot

import keras
import torch
import numpy as np
import albumentations as A

from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from agx_core.transforms import LogTransform, Deskew


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir: Path, cond_shape: Sequence[int], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.cond_shape = cond_shape
        # Get list of all image file names in the folder
        self.image_files = list(root_dir.glob("*.bmp"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert("L")

        if self.transform:
            image = self.transform(image=np.array(image))
            image = image["image"][..., np.newaxis]

        condition = np.ones(self.cond_shape, dtype=np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = torch.tensor(image, device=device)
        return (image, torch.tensor(condition, device=device)), image


def train_transforms(img_size, mean=[0.5], std=[0.5]):
    return A.Compose(
        [
            Deskew(),
            # A.Pad((25, 25), 255),
            LogTransform(epsilon=1, invert=True),
            A.Resize(img_size, img_size),
            A.Affine(scale=(0.9, 0.95), rotate=(-90, 90), shear=(5, 5), p=0.5),
            A.RandomRotate90(0.5),
            A.GaussianBlur(blur_range=(1, 3), p=0.3),
            A.Normalize(mean=mean, std=std),
        ]
    )


def valid_transforms(img_size, mean=[0.5], std=[0.5]):
    return A.Compose(
        [
            Deskew(),
            # A.Pad((25, 25), 255),
            LogTransform(epsilon=1, invert=True),
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
        ]
    )


# %%
img_size = 224
res = img_size // 2**5

img_shape = (None, img_size, img_size, 1)
cond_shape = (None, res, res, 1)

# %%
train_path = Path("./data/products/LaTuaPastaGlassJars/Clean/train/")
valid_path = Path("./data/products/LaTuaPastaGlassJars/Clean/val/")

ds_train = UnlabeledImageDataset(
    train_path, transform=train_transforms(img_size), cond_shape=cond_shape[1:]
)
ds_valid = UnlabeledImageDataset(
    valid_path, transform=valid_transforms(img_size), cond_shape=cond_shape[1:]
)

# %%
from keras.optimizers import Adam

from agx_core.models.reversed_autoencoder import (
    MobileNetV3SmallEncoder,
    MobileNetV3SmallDecoder,
)
from agx_core.models.reversed_autoencoder.model import mse_weighted
from agx_torch.models.reversed_autoencoder.model import ReversedAutoencoder

spatial_temperature = 8.0

ra_model = Path("ra_mbnetv3.model.keras")
if ra_model.exists():
    ra: ReversedAutoencoder = keras.models.load_model(ra_model)
    ra.place_on_devices("cuda:0", "cuda:1")
else:
    enc = MobileNetV3SmallEncoder(latent_size=512, progressive=True)
    dec = MobileNetV3SmallDecoder(target_shape=img_shape[1:], progressive=True)
    ra = ReversedAutoencoder(
        enc,
        dec,
        beta_kld=0.1,
        spatial_temperature=spatial_temperature,
        freeze_backbone=False,
    )
    ra.build([img_shape, cond_shape])
    ra.place_on_devices("cuda:0", "cuda:1")
    ra.compile(Adam(learning_rate=5e-7), Adam(learning_rate=1e-4))

ra.summary()

loader_train = DataLoader(ds_train, batch_size=10, shuffle=True)
loader_valid = DataLoader(ds_valid, batch_size=10)

# %%
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

rec_dir = Path("reconstructions")
rec_dir.mkdir(exist_ok=True)

plot_every = 25
rec_test_samples = 10
(samples, labels), _ = next(iter(loader_valid))


def plot_on_step_end(step: int, logs: Optional[Dict[str, Any]] = None):
    # plot reconstruction every n epochs/batches
    if (step + 1) % plot_every != 0:
        return

    with torch.no_grad():
        reconstructed = ra([samples, labels])
        samples_resized = ra.resize_progressive_output(samples)
        error = mse_weighted(samples_resized, reconstructed, spatial_temperature)

    samples_resized = samples_resized.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    error = error.cpu().numpy()

    height, width = samples_resized.shape[1:3]
    ar = width / height

    figwidth = 4
    figheight = figwidth / ar

    # 3 columns: original, reconstructed, difference
    fig, axs = plt.subplots(
        rec_test_samples,
        3,
        figsize=(3 * figwidth, rec_test_samples * figheight),
    )

    # Handle single row case
    if rec_test_samples == 1:
        axs = axs.reshape(1, -1)

    for row, (real, rec, err) in enumerate(zip(samples_resized, reconstructed, error)):

        if row == 0:
            axs[row, 0].set_title("Original", fontsize=12, pad=15)
            axs[row, 1].set_title("Reconstruction", fontsize=12, pad=15)
            axs[row, 2].set_title("Anomaly Map", fontsize=12, pad=15)

        # First column: keep axis on for ylabel, hide ticks, rotate 90 degrees
        axs[row, 0].set_ylabel(
            f"{row+1}",
            fontsize=10,
            rotation=90,
            labelpad=15,
            ha="center",
            va="center",
        )

        # All columns: hide ticks
        for col in range(3):
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])

        axs[row, 0].imshow(real, cmap="gray")
        axs[row, 1].imshow(rec, cmap="gray")
        axs[row, 2].imshow(err, cmap="inferno")

    epoch_num = step + 1
    filename = f"epoch_{epoch_num:05d}.jpg"
    title = f"Reconstruction Results - Epoch {epoch_num}"

    # Add overall figure title
    fig.suptitle(title, fontsize=14, y=0.96)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.08)

    fig.savefig(rec_dir / filename)

    plt.close(fig)

    print(f"Epoch {step + 1}: reconstruction results saved")


# %%
from keras.callbacks import ModelCheckpoint, LambdaCallback
from agx_core.models.reversed_autoencoder.callbacks import (
    AdversarialEquilibriumCallback,
    ProgressiveGrowingCallback,
)

callbacks = [
    AdversarialEquilibriumCallback(0.5, -0.5, min_pause_steps=200),
    ProgressiveGrowingCallback(20000, 20000),
    ModelCheckpoint(
        filepath="ra_mbnetv3.best.keras",
        monitor="val_loss_rec",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    LambdaCallback(on_epoch_end=plot_on_step_end),
]

# %%
history = ra.fit(
    loader_train,
    validation_data=loader_valid,
    epochs=10000,
    callbacks=callbacks,
    verbose=2,
)

# %%
ra.save("ra_mbnetv3.model.keras")

# %%
import pandas as pd

df = pd.DataFrame.from_dict(history.history)

hist_file = Path("history.csv")
if hist_file.exists():
    hist = pd.read_csv(hist_file)
    df.index += len(hist)
    hist = pd.concat([hist, df])
    hist.to_csv(hist_file, index=False)
else:
    df.to_csv(hist_file, index=False)
    hist = df
