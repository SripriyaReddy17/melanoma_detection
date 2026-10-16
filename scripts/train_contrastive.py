"""
Train Cross-Modal Contrastive Learning module for CEFM.

Uses:
    HAM10000 train.csv       -> training
    HAM10000 validation.csv  -> validation
    HAM10000 test.csv        -> kept untouched

Each pair contains:
    ViT image embedding: 768-D
    ABCDE clinical vector: 16-D

Synthetic features are used only with --smoke-test.
"""

import os
import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from app.ml.contrastive.model import (
    CEFMContrastiveModel,
    nt_xent_loss,
)

from app.services.classification import get_vit_embedding
from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.abcde.vector import build_abcde_vector


# ============================================================
# Paths
# ============================================================

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "Ham10000"
    / "processed"
)

IMAGE_DIRS = [
    BASE_DIR
    / "datasets"
    / "Ham10000"
    / "HAM10000_images_part_1",

    BASE_DIR
    / "datasets"
    / "Ham10000"
    / "HAM10000_images_part_2",
]

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "contrastive"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR = (
    BASE_DIR
    / "outputs"
    / "contrastive"
    / "cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT = (
    OUTPUT_DIR
    / "contrastive_projection_best.pt"
)

LAST_CHECKPOINT = (
    OUTPUT_DIR
    / "contrastive_projection_last.pt"
)

HISTORY_FILE = (
    BASE_DIR
    / "outputs"
    / "contrastive"
    / "contrastive_training_history.json"
)


# ============================================================
# Dataset
# ============================================================

class ContrastiveDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        image_embedding, clinical_vector = self.samples[idx]

        return (
            torch.tensor(
                image_embedding,
                dtype=torch.float32
            ),
            torch.tensor(
                clinical_vector,
                dtype=torch.float32
            ),
        )


# ============================================================
# Image map
# ============================================================

def build_image_map():

    disk_map = {}

    for directory in IMAGE_DIRS:

        if not directory.exists():
            continue

        for image_path in directory.glob("*.jpg"):

            disk_map[
                image_path.stem
            ] = image_path

    print(
        f"[Contrastive Training] "
        f"Found {len(disk_map)} HAM10000 images."
    )

    return disk_map


# ============================================================
# Feature extraction
# ============================================================

def extract_real_pairs(
    csv_path,
    max_samples=None,
    split_name="train",
):

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n{csv_path}"
        )

    print(
        f"\n[Contrastive Training] "
        f"Loading {split_name} split:"
    )

    print(csv_path)

    df = pd.read_csv(csv_path)

    print(
        f"[Contrastive Training] "
        f"{split_name} rows: {len(df)}"
    )

    image_map = build_image_map()

    pairs = []

    processed = 0

    for _, row in df.iterrows():

        if (
            max_samples is not None
            and len(pairs) >= max_samples
        ):
            break

        image_id = str(
            row["image_id"]
        )

        if image_id not in image_map:

            print(
                f"[Contrastive Training] "
                f"Image not found: {image_id}"
            )

            continue

        image_path = image_map[image_id]

        processed += 1

        print(
            f"[Contrastive Training] "
            f"{split_name}: "
            f"{len(pairs) + 1}"
            f"{'/' + str(max_samples) if max_samples else ''}"
            f" -> {image_id}"
        )

        try:

            # ------------------------------------------------
            # ViT embedding
            # ------------------------------------------------

            embedding = get_vit_embedding(
                image_path
            )

            if embedding is None:

                print(
                    f"  Skipped: "
                    f"ViT embedding unavailable"
                )

                continue

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

            if embedding.shape[0] != 768:

                print(
                    f"  Skipped: expected "
                    f"768-D embedding, got "
                    f"{embedding.shape[0]}"
                )

                continue

            # ------------------------------------------------
            # Load image
            # ------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                print(
                    "  Skipped: image could not "
                    "be loaded"
                )

                continue

            # ------------------------------------------------
            # Segmentation
            # ------------------------------------------------

            mask = segment_lesion(
                str(image_path)
            )

            if mask is None:

                print(
                    "  Skipped: segmentation "
                    "failed"
                )

                continue

            # ------------------------------------------------
            # ABCDE extraction
            # ------------------------------------------------

            features = extract_abcde_features(
                image,
                mask
            )

            if features is None:

                print(
                    "  Skipped: ABCDE extraction "
                    "failed"
                )

                continue

            # ------------------------------------------------
            # Build 16-D ABCDE vector
            # ------------------------------------------------

            abcde_vector = build_abcde_vector(
                features
            )

            clinical_vector = np.asarray(
                abcde_vector.to_numpy(),
                dtype=np.float32
            ).reshape(-1)

            if clinical_vector.shape[0] != 16:

                print(
                    f"  Skipped: expected "
                    f"16-D ABCDE vector, got "
                    f"{clinical_vector.shape[0]}"
                )

                continue

            pairs.append(
                (
                    embedding,
                    clinical_vector
                )
            )

        except Exception as e:

            print(
                f"  Skipped due to error: {e}"
            )

    print(
        f"\n[Contrastive Training] "
        f"{split_name} real pairs collected: "
        f"{len(pairs)}"
    )

    if len(pairs) < 4:

        raise RuntimeError(
            f"Only {len(pairs)} real pairs were "
            f"collected for {split_name}. "
            f"At least 4 are required."
        )

    return pairs


# ============================================================
# Cache helpers
# ============================================================

def cache_path(split_name):

    return (
        CACHE_DIR
        / f"{split_name}_contrastive_pairs.npz"
    )


def save_pairs_to_cache(
    pairs,
    split_name
):

    image_embeddings = np.stack(
        [p[0] for p in pairs]
    )

    clinical_vectors = np.stack(
        [p[1] for p in pairs]
    )

    path = cache_path(
        split_name
    )

    np.savez_compressed(
        path,
        image_embeddings=image_embeddings,
        clinical_vectors=clinical_vectors,
    )

    print(
        f"[Contrastive Training] "
        f"Cached {len(pairs)} {split_name} pairs:"
    )

    print(path)


def load_pairs_from_cache(
    split_name
):

    path = cache_path(
        split_name
    )

    if not path.exists():

        return None

    try:

        data = np.load(
            path
        )

        image_embeddings = data[
            "image_embeddings"
        ]

        clinical_vectors = data[
            "clinical_vectors"
        ]

        if (
            len(image_embeddings)
            != len(clinical_vectors)
        ):

            print(
                f"[Contrastive Training] "
                f"Invalid cache for {split_name}."
            )

            return None

        pairs = list(
            zip(
                image_embeddings.astype(
                    np.float32
                ),
                clinical_vectors.astype(
                    np.float32
                ),
            )
        )

        print(
            f"[Contrastive Training] "
            f"Loaded {len(pairs)} "
            f"{split_name} pairs from cache."
        )

        return pairs

    except Exception as e:

        print(
            f"[Contrastive Training] "
            f"Could not load cache: {e}"
        )

        return None

def get_pairs(
    split_name,
    max_samples=None,
    use_cache=True,
):
    cached_pairs = None

    if use_cache:
        cached_pairs = load_pairs_from_cache(split_name)

        if cached_pairs is not None:
            cached_count = len(cached_pairs)

            if max_samples is None:
                return cached_pairs

            if cached_count >= max_samples:
                print(
                    f"[Contrastive Training] "
                    f"Cache has {cached_count} {split_name} pairs; "
                    f"using first {max_samples}."
                )
                return cached_pairs[:max_samples]

            print(
                f"[Contrastive Training] "
                f"Cache has only {cached_count} {split_name} pairs, "
                f"but {max_samples} requested. "
                f"Regenerating cache."
            )

    csv_path = DATASET_DIR / f"{split_name}.csv"

    pairs = extract_real_pairs(
        csv_path=csv_path,
        max_samples=max_samples,
        split_name=split_name,
    )

    if use_cache:
        save_pairs_to_cache(
            pairs,
            split_name
        )

    return pairs

# ============================================================
# Smoke-test pairs
# ============================================================

def create_smoke_pairs(
    num_samples=16
):

    print(
        "[Contrastive Training] "
        "SMOKE TEST MODE"
    )

    pairs = []

    for _ in range(
        num_samples
    ):

        image_embedding = (
            np.random.randn(
                768
            ).astype(
                np.float32
            )
        )

        clinical_vector = (
            np.random.uniform(
                0.1,
                0.9,
                size=16
            ).astype(
                np.float32
            )
        )

        pairs.append(
            (
                image_embedding,
                clinical_vector
            )
        )

    return pairs


# ============================================================
# One epoch
# ============================================================

def run_epoch(
    model,
    dataloader,
    optimizer,
    device,
    training=True,
):

    if training:

        model.train()

    else:

        model.eval()

    total_loss = 0.0
    steps = 0

    for image_embeddings, clinical_vectors in dataloader:

        image_embeddings = (
            image_embeddings.to(device)
        )

        clinical_vectors = (
            clinical_vectors.to(device)
        )

        with torch.set_grad_enabled(
            training
        ):

            image_projection, clinical_projection = model(
                image_embeddings,
                clinical_vectors
            )

            loss = nt_xent_loss(
                image_projection,
                clinical_projection,
                temperature=model.temperature
            )

            if training:

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        total_loss += loss.item()

        steps += 1

    return (
        total_loss
        / max(1, steps)
    )


# ============================================================
# Training
# ============================================================

def train_contrastive(
    epochs=5,
    batch_size=16,
    lr=1e-4,
    train_samples=500,
    validation_samples=200,
    smoke_test=False,
    use_cache=True,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n[Contrastive Training] "
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    if smoke_test:

        train_pairs = create_smoke_pairs(
            16
        )

        validation_pairs = create_smoke_pairs(
            16
        )

    else:

        train_pairs = get_pairs(
            split_name="train",
            max_samples=train_samples,
            use_cache=use_cache,
        )

        validation_pairs = get_pairs(
            split_name="validation",
            max_samples=validation_samples,
            use_cache=use_cache,
        )

    print(
        f"\nTraining pairs: "
        f"{len(train_pairs)}"
    )

    print(
        f"Validation pairs: "
        f"{len(validation_pairs)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = ContrastiveDataset(
        train_pairs
    )

    validation_dataset = ContrastiveDataset(
        validation_pairs
    )

    train_batch_size = min(
        batch_size,
        len(train_dataset)
    )

    validation_batch_size = min(
        batch_size,
        len(validation_dataset)
    )

    if train_batch_size < 2:

        raise RuntimeError(
            "Training requires at least "
            "2 samples per batch."
        )

    if validation_batch_size < 2:

        raise RuntimeError(
            "Validation requires at least "
            "2 samples per batch."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        drop_last=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CEFMContrastiveModel(
        image_dim=768,
        clinical_dim=16,
        latent_dim=128,
    ).to(device)

    print(
        "\n[Contrastive Training] "
        "Model configuration:"
    )

    print(
        "  Image dimension: 768"
    )

    print(
        "  ABCDE dimension: 16"
    )

    print(
        "  Projection dimension: 128"
    )

    print(
        f"  Temperature: "
        f"{model.temperature}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            training=True,
        )

        validation_loss = run_epoch(
            model=model,
            dataloader=validation_loader,
            optimizer=None,
            device=device,
            training=False,
        )

        record = {
            "epoch": epoch,
            "train_loss": float(
                train_loss
            ),
            "validation_loss": float(
                validation_loss
            ),
        }

        history.append(
            record
        )

        print(
            f"\nEpoch [{epoch}/{epochs}]"
        )

        print(
            f"  Train NT-Xent Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Validation NT-Xent Loss: "
            f"{validation_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT
            )

            print(
                "  ✓ New best validation "
                "checkpoint saved."
            )

    # --------------------------------------------------------
    # Save final checkpoint
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        LAST_CHECKPOINT
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_output = {
        "device": str(device),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(
            validation_pairs
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_validation_loss": (
            float(best_validation_loss)
        ),
        "history": history,
    }

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history_output,
            f,
            indent=2
        )

    print(
        "\n===================================="
    )

    print(
        "Contrastive training completed."
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Best checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print(
        f"\nLast checkpoint:"
    )

    print(
        LAST_CHECKPOINT
    )

    print(
        f"\nTraining history:"
    )

    print(
        HISTORY_FILE
    )

    print(
        "===================================="
    )

    return BEST_CHECKPOINT


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train CEFM cross-modal "
            "contrastive projection heads."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
    )

    args = parser.parse_args()

    smoke_test = (
        args.smoke_test
        or os.getenv(
            "SMOKE_TEST"
        ) == "true"
    )

    train_contrastive(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        smoke_test=smoke_test,
        use_cache=not args.no_cache,
    )