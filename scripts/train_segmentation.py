"""Legacy ISIC2018 UNet training experiment; not active app segmentation.

This script writes ``models/segmentation/unet_isic2018.pt``. Application
runtime uses the separately managed UltraLight VM-UNet checkpoint and this
script must not be used to replace it.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from app.ml.segmentation.unet import UNet


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "ISIC2018"

TRAIN_IMG_DIR = (
    DATASET_DIR /
    "ISIC2018_Task1-2_Training_Input"
)

TRAIN_MASK_DIR = (
    DATASET_DIR /
    "ISIC2018_Task1_Training_GroundTruth"
)

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "segmentation"
)

OUTPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "segmentation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    MODEL_DIR /
    "unet_isic2018.pt"
)

SPLIT_PATH = (
    OUTPUT_DIR /
    "segmentation_split.json"
)

HISTORY_PATH = (
    OUTPUT_DIR /
    "segmentation_training_history.json"
)


# ============================================================
# DATASET
# ============================================================

class ISICSegmentationDataset(Dataset):

    def __init__(
        self,
        image_paths,
        mask_paths,
        img_size=256
    ):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (
                self.img_size,
                self.img_size
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        mask = cv2.resize(
            mask,
            (
                self.img_size,
                self.img_size
            ),
            interpolation=cv2.INTER_NEAREST
        )

        # ----------------------------------------------------
        # Image tensor
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float() / 255.0

        # ----------------------------------------------------
        # Mask tensor
        # ----------------------------------------------------

        mask = (
            mask > 127
        ).astype(np.float32)

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    logits,
    target,
    smooth=1.0
):

    prediction = torch.sigmoid(
        logits
    )

    prediction = prediction.view(-1)

    target = target.view(-1)

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2.0 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )

    return 1.0 - dice


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(
    logits,
    target,
    threshold=0.5
):

    prediction = (
        torch.sigmoid(logits)
        >= threshold
    ).float()

    target = (
        target > 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum(dim=(1, 2, 3))

    denominator = (
        prediction.sum(dim=(1, 2, 3))
        +
        target.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + 1e-7
    ) / (
        denominator + 1e-7
    )

    return dice.mean().item()


# ============================================================
# FIND IMAGE/MASK PAIRS
# ============================================================

def discover_pairs():

    if not TRAIN_IMG_DIR.exists():

        raise FileNotFoundError(
            f"Training image directory not found:\n"
            f"{TRAIN_IMG_DIR}"
        )

    if not TRAIN_MASK_DIR.exists():

        raise FileNotFoundError(
            f"Training mask directory not found:\n"
            f"{TRAIN_MASK_DIR}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only image extensions are accepted.
    # LICENSE.txt will automatically be ignored.
    # --------------------------------------------------------

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    images = sorted(
        [
            p
            for p in TRAIN_IMG_DIR.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    masks = sorted(
        [
            p
            for p in TRAIN_MASK_DIR.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    # --------------------------------------------------------
    # Create mask lookup
    # --------------------------------------------------------

    mask_map = {}

    for mask in masks:

        name = mask.stem

        if name.endswith(
            "_segmentation"
        ):

            name = name[
                :-13
            ]

        mask_map[name] = mask

    # --------------------------------------------------------
    # Match images with masks
    # --------------------------------------------------------

    pairs = []

    for image in images:

        if image.stem in mask_map:

            pairs.append(
                (
                    image,
                    mask_map[image.stem]
                )
            )

    print(
        f"\nFound {len(pairs)} image-mask pairs."
    )

    if len(pairs) == 0:

        raise RuntimeError(
            "No image-mask pairs found."
        )

    return pairs


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_split(
    pairs,
    validation_ratio,
    seed
):

    random_generator = random.Random(
        seed
    )

    indices = list(
        range(len(pairs))
    )

    random_generator.shuffle(
        indices
    )

    validation_count = int(
        len(indices)
        * validation_ratio
    )

    validation_count = max(
        1,
        validation_count
    )

    validation_indices = sorted(
        indices[
            :validation_count
        ]
    )

    training_indices = sorted(
        indices[
            validation_count:
        ]
    )

    return (
        training_indices,
        validation_indices
    )


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(
    pairs,
    training_indices,
    validation_indices,
    seed,
    validation_ratio
):

    data = {

        "dataset":
            "ISIC2018",

        "seed":
            seed,

        "validation_ratio":
            validation_ratio,

        "training_count":
            len(training_indices),

        "validation_count":
            len(validation_indices),

        "training_images":
        [
            pairs[i][0].name
            for i in training_indices
        ],

        "validation_images":
        [
            pairs[i][0].name
            for i in validation_indices
        ]
    }

    with open(
        SPLIT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    device
):

    model.eval()

    losses = []

    dice_scores = []

    bce_loss = (
        nn.BCEWithLogitsLoss()
    )

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            logits = model(
                images
            )

            loss = (
                bce_loss(
                    logits,
                    masks
                )
                +
                dice_loss(
                    logits,
                    masks
                )
            )

            losses.append(
                loss.item()
            )

            dice_scores.append(
                dice_score(
                    logits,
                    masks
                )
            )

    return (
        float(np.mean(losses)),
        float(np.mean(dice_scores))
    )


# ============================================================
# TRAIN
# ============================================================

def train(
    epochs=20,
    batch_size=8,
    learning_rate=1e-4,
    validation_ratio=0.20,
    seed=42,
    image_size=256
):

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n=========================================="
    )

    print(
        "UNET ISIC2018 SEGMENTATION TRAINING"
    )

    print(
        "=========================================="
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "WARNING: CUDA is not available."
        )

    print(
        f"Dataset: {DATASET_DIR}"
    )

    print(
        f"Validation split: "
        f"{validation_ratio * 100:.0f}%"
    )

    print(
        f"Random seed: {seed}"
    )

    # --------------------------------------------------------
    # Find pairs
    # --------------------------------------------------------

    pairs = discover_pairs()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        training_indices,
        validation_indices
    ) = create_split(
        pairs,
        validation_ratio,
        seed
    )

    save_split(
        pairs,
        training_indices,
        validation_indices,
        seed,
        validation_ratio
    )

    training_pairs = [
        pairs[i]
        for i in training_indices
    ]

    validation_pairs = [
        pairs[i]
        for i in validation_indices
    ]

    print(
        f"\nTraining samples: "
        f"{len(training_pairs)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_pairs)}"
    )

    print(
        f"\nSplit saved to:"
        f"\n{SPLIT_PATH}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    training_dataset = (
        ISICSegmentationDataset(
            [p[0] for p in training_pairs],
            [p[1] for p in training_pairs],
            image_size
        )
    )

    validation_dataset = (
        ISICSegmentationDataset(
            [p[0] for p in validation_pairs],
            [p[1] for p in validation_pairs],
            image_size
        )
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    # Windows is more reliable with 0 workers
    # while debugging dataset loading.

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_features=16
    ).to(device)

    print(
        "\nUNet model created."
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    bce_loss = (
        nn.BCEWithLogitsLoss()
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_dice = -1.0

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        training_losses = []

        for (
            images,
            masks
        ) in training_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            loss = (
                bce_loss(
                    logits,
                    masks
                )
                +
                dice_loss(
                    logits,
                    masks
                )
            )

            loss.backward()

            optimizer.step()

            training_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(
                training_losses
            )
        )

        validation_loss, validation_dice = (
            validate(
                model,
                validation_loader,
                device
            )
        )

        history.append({

            "epoch": epoch,

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            "validation_dice":
                validation_dice

        })

        print(
            f"\nEpoch "
            f"[{epoch}/{epochs}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Dice: "
            f"{validation_dice:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            validation_dice
            >
            best_validation_dice
        ):

            best_validation_dice = (
                validation_dice
            )

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print(
                "\n*** NEW BEST MODEL ***"
            )

            print(
                f"Best Validation Dice: "
                f"{best_validation_dice:.4f}"
            )

            print(
                f"Saved to:"
                f"\n{CHECKPOINT_PATH}"
            )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    print(
        "\n=========================================="
    )

    print(
        "TRAINING FINISHED"
    )

    print(
        "=========================================="
    )

    print(
        f"Best Validation Dice: "
        f"{best_validation_dice:.4f}"
    )

    print(
        f"Model:"
        f"\n{CHECKPOINT_PATH}"
    )

    print(
        f"History:"
        f"\n{HISTORY_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=256
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        validation_ratio=args.val_split,
        seed=args.seed,
        image_size=args.img_size
    )