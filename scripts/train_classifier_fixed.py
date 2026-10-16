from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_CSV = BASE_DIR / "datasets" / "Ham10000" / "processed" / "train.csv"
VAL_CSV = BASE_DIR / "datasets" / "Ham10000" / "processed" / "validation.csv"
TEST_CSV = BASE_DIR / "datasets" / "Ham10000" / "processed" / "test.csv"
SPLIT_METADATA = (
    BASE_DIR / "datasets" / "Ham10000" / "processed" / "split_metadata.json"
)
LIVE_MODEL_DIR = BASE_DIR / "models" / "classification" / "vit_model_experiment"
LIVE_CHECKPOINT = LIVE_MODEL_DIR / "model.safetensors"

MODEL_BACKBONE = "google/vit-base-patch16-224"
IMAGE_SIZE = 224
MODEL_NAME = "ViT-Melanoma"
MODEL_SELECTION_METRIC = "validation_pr_auc"

# This candidate directory is deliberately not used by application inference.
OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "classification"
    / "vit_candidate_run_001"
)
RUN_METADATA_PATH = OUTPUT_DIR / "run_metadata.json"


MANIFEST_COLUMNS = {
    "split",
    "image_id",
    "image_path",
    "lesion_id",
    "label",
    "label_name",
}


def resolve_image_path(image_path: str) -> Path:
    """Resolve the manifest's authoritative image_path."""
    path = Path(image_path)
    resolved = path if path.is_absolute() else BASE_DIR / path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Image listed in manifest does not exist: {resolved}"
        )
    return resolved


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open("rb") as file_obj:
        return hashlib.file_digest(file_obj, "sha256").hexdigest()


def load_split(csv_path: Path, expected_split: str) -> pd.DataFrame:
    """Load and validate one existing fixed split manifest."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    missing = MANIFEST_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError(f"{csv_path.name} contains no manifest rows.")
    if df["split"].isna().any() or not df["split"].eq(expected_split).all():
        raise ValueError(
            f"{csv_path.name} must contain only split={expected_split!r} rows."
        )
    if df["image_id"].isna().any() or df["image_id"].duplicated().any():
        raise ValueError(f"{csv_path.name} has missing or duplicate image IDs.")
    if df["lesion_id"].isna().any() or df["lesion_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{csv_path.name} has missing lesion IDs.")

    labels = pd.to_numeric(df["label"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"{csv_path.name} labels must be binary 0/1.")
    df["label"] = labels.astype(int)

    expected_names = df["label"].map({0: "non_mel", 1: "mel"})
    if df["label_name"].isna().any() or not df["label_name"].eq(expected_names).all():
        raise ValueError(
            f"{csv_path.name} label_name values do not match binary labels."
        )
    if not df["label"].isin([0, 1]).all() or set(df["label"]) != {0, 1}:
        raise ValueError(
            f"{csv_path.name} must include both binary classes 0 and 1."
        )

    df["image_path"] = df["image_path"].astype(str)
    for image_path in df["image_path"]:
        resolve_image_path(image_path)

    return df.reset_index(drop=True)


def validate_experiment_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fail closed unless manifests and source match split provenance."""
    train_df = load_split(TRAIN_CSV, expected_split="train")
    val_df = load_split(VAL_CSV, expected_split="validation")
    test_df = load_split(TEST_CSV, expected_split="test")

    lesion_overlaps = {
        "train_validation": set(train_df["lesion_id"]) & set(val_df["lesion_id"]),
        "train_test": set(train_df["lesion_id"]) & set(test_df["lesion_id"]),
        "validation_test": set(val_df["lesion_id"]) & set(test_df["lesion_id"]),
    }
    if any(lesion_overlaps.values()):
        raise ValueError(
            "Lesion overlap detected in fixed manifests: "
            + ", ".join(
                f"{name}={len(overlap)}"
                for name, overlap in lesion_overlaps.items()
            )
        )

    image_overlaps = {
        "train_validation": set(train_df["image_id"]) & set(val_df["image_id"]),
        "train_test": set(train_df["image_id"]) & set(test_df["image_id"]),
        "validation_test": set(val_df["image_id"]) & set(test_df["image_id"]),
    }
    if any(image_overlaps.values()):
        raise ValueError(
            "Image ID overlap detected in fixed manifests: "
            + ", ".join(
                f"{name}={len(overlap)}"
                for name, overlap in image_overlaps.items()
            )
        )

    if not SPLIT_METADATA.is_file():
        raise FileNotFoundError(
            f"Split provenance file not found: {SPLIT_METADATA}"
        )
    provenance = json.loads(SPLIT_METADATA.read_text(encoding="utf-8"))
    validation_checks = provenance.get("validation_checks", {})
    if not isinstance(validation_checks, dict) or not validation_checks:
        raise ValueError(
            "Split provenance does not contain validation check results."
        )
    failed_checks = [
        name
        for name, check in validation_checks.items()
        if not isinstance(check, dict) or check.get("passed") is not True
    ]
    if failed_checks:
        raise ValueError(
            "Split provenance contains failed or incomplete validation checks: "
            + ", ".join(sorted(failed_checks))
        )
    manifest_paths = {
        "train": TRAIN_CSV,
        "validation": VAL_CSV,
        "test": TEST_CSV,
    }
    expected_hashes = provenance.get("manifest_sha256", {})
    manifest_hashes = {}
    for split_name, manifest_path in manifest_paths.items():
        expected_hash = expected_hashes.get(split_name)
        if not expected_hash and split_name == "test":
            expected_hash = provenance.get("test_manifest_sha256")
        if not expected_hash:
            raise ValueError(
                f"Split provenance does not record the {split_name} manifest hash."
            )
        actual_hash = sha256_file(manifest_path)
        if actual_hash.lower() != str(expected_hash).lower():
            raise ValueError(
                f"{split_name} manifest hash does not match split provenance."
            )
        manifest_hashes[split_name] = actual_hash

    source_metadata = BASE_DIR / provenance["source_metadata_filename"]
    if not source_metadata.is_file():
        raise FileNotFoundError(
            f"Source metadata file not found: {source_metadata}"
        )
    source_hash = sha256_file(source_metadata)
    if source_hash.lower() != str(
        provenance.get("source_metadata_sha256", "")
    ).lower():
        raise ValueError("Source metadata hash does not match split provenance.")

    recorded_splits = provenance.get("splits", {})
    for split_name, frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        recorded = recorded_splits.get(split_name, {})
        actual_class_counts = {
            "non_mel": int((frame["label"] == 0).sum()),
            "mel": int((frame["label"] == 1).sum()),
        }
        actual_counts = {
            "image_count": len(frame),
            "lesion_count": frame["lesion_id"].nunique(),
            "class_counts": actual_class_counts,
        }
        for key, value in actual_counts.items():
            if recorded.get(key) != value:
                raise ValueError(
                    f"{split_name} {key} does not match split provenance."
                )

    if not LIVE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Live application checkpoint is missing: {LIVE_CHECKPOINT}"
        )
    candidate_path = OUTPUT_DIR.resolve()
    live_path = LIVE_MODEL_DIR.resolve()
    if candidate_path == live_path or candidate_path.is_relative_to(live_path):
        raise ValueError(
            "Candidate checkpoint directory must be separate from the live model."
        )
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Candidate output already exists; refusing to overwrite: {OUTPUT_DIR}"
        )

    provenance["verified_manifest_sha256"] = manifest_hashes
    provenance["verified_source_metadata_sha256"] = source_hash
    provenance["live_checkpoint_sha256_before_training"] = sha256_file(
        LIVE_CHECKPOINT
    )
    provenance["lesion_overlap_counts"] = {
        name: len(overlap) for name, overlap in lesion_overlaps.items()
    }
    provenance["image_id_overlap_counts"] = {
        name: len(overlap) for name, overlap in image_overlaps.items()
    }
    return train_df, val_df, provenance


def build_weighted_criterion(labels, device):
    counts = np.bincount(
        np.asarray(labels, dtype=np.int64),
        minlength=2,
    ).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("Training split must contain both classes.")
    class_weights = np.max(counts) / counts
    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )
    return torch.nn.CrossEntropyLoss(weight=weights), class_weights.tolist()


def make_shuffled_loader(dataset, batch_size, collate_fn):
    """Build the ordinary shuffled loader; no weighted sampler is used."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )


def compute_validation_metrics(labels, predictions, melanoma_probabilities):
    """Compute baseline argmax metrics and probability-based AUC metrics."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    melanoma_probabilities = np.asarray(
        melanoma_probabilities,
        dtype=np.float64,
    )
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            "Validation PR-AUC requires both binary classes in the labels."
        )
    if not np.isfinite(melanoma_probabilities).all():
        raise ValueError("Validation melanoma probabilities must be finite.")

    tn, fp, _, _ = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    return {
        "sensitivity_recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "roc_auc": float(roc_auc_score(labels, melanoma_probabilities)),
        "pr_auc": float(
            average_precision_score(labels, melanoma_probabilities)
        ),
    }


def build_run_metadata(args, provenance, run_id):
    return {
        "experiment_id": run_id,
        "model_name": MODEL_NAME,
        "backbone": MODEL_BACKBONE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "optimizer": "AdamW",
        "max_steps_per_epoch": int(args.max_steps),
        "freeze_backbone": bool(args.freeze_backbone),
        "loss": {
            "name": "weighted_cross_entropy",
            "class_weights": None,
        },
        "sampler": {
            "name": "ordinary_shuffle",
            "weighted_random_sampler": False,
        },
        "scheduler": None,
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "decision_threshold": "argmax (equivalent to 0.5 for two classes)",
        "manifest_sha256": provenance["verified_manifest_sha256"],
        "source_metadata_sha256": provenance[
            "verified_source_metadata_sha256"
        ],
        "source_metadata_filename": provenance["source_metadata_filename"],
        "label_definition": provenance["label_definition"],
        "lesion_overlap_counts": provenance["lesion_overlap_counts"],
        "image_id_overlap_counts": provenance["image_id_overlap_counts"],
        "validation_checks_passed": True,
        "live_checkpoint_sha256_before_training": provenance[
            "live_checkpoint_sha256_before_training"
        ],
        "history": [],
        "best_checkpoint": None,
    }


def save_run_metadata(metadata):
    """Atomically persist run configuration and epoch history."""
    temporary_path = RUN_METADATA_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(RUN_METADATA_PATH)


def is_better_checkpoint(validation_pr_auc, best_pr_auc):
    if not np.isfinite(validation_pr_auc):
        raise ValueError("Validation PR-AUC must be finite for model selection.")
    return validation_pr_auc > best_pr_auc


class HAMDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        augment: bool = False,
    ):
        self.frame = frame
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        row = self.frame.iloc[idx]

        image_path = resolve_image_path(str(row["image_path"]))

        image = Image.open(image_path).convert("RGB")

        if self.augment:

            if np.random.rand() < 0.5:
                image = ImageOps.mirror(image)

            if np.random.rand() < 0.5:
                image = image.rotate(
                    np.random.uniform(-20, 20),
                    resample=Image.BILINEAR,
                )

            if np.random.rand() < 0.25:
                image = ImageOps.autocontrast(image)

        label = int(row["label"])

        return image, label


def build_model():

    model = ViTForImageClassification.from_pretrained(
        MODEL_BACKBONE
    )

    model.config.num_labels = 2

    model.config.id2label = {
        0: "non_mel",
        1: "mel",
    }

    model.config.label2id = {
        "non_mel": 0,
        "mel": 1,
    }

    model.num_labels = 2

    model.classifier = torch.nn.Linear(
        model.config.hidden_size,
        2,
    )

    model.config.problem_type = (
        "single_label_classification"
    )

    return model


def train(args):

    train_df, val_df, provenance = validate_experiment_inputs()

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_metadata = build_run_metadata(args, provenance, run_id)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    save_run_metadata(run_metadata)

    print("=" * 70)
    print("CEFM — ViT Fixed-Split Training")
    print("=" * 70)
    print(f"Experiment: {run_id}")
    print(f"Candidate checkpoint: {OUTPUT_DIR}")
    print(f"Model selection: {MODEL_SELECTION_METRIC}")

    print("\nDataset:")
    print(f"Train images:      {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images reserved for final evaluation: {provenance['splits']['test']['image_count']}")

    print("\nTrain distribution:")
    print(train_df["label"].value_counts().sort_index())
    print("\nValidation distribution:")
    print(val_df["label"].value_counts().sort_index())

    processor = ViTImageProcessor.from_pretrained(MODEL_BACKBONE)
    train_dataset = HAMDataset(train_df, augment=True)
    val_dataset = HAMDataset(val_df, augment=False)

    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    train_loader = make_shuffled_loader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = build_model()
    if args.freeze_backbone:
        print("Freezing ViT backbone; training classifier head only.")
        for param in model.vit.parameters():
            param.requires_grad = False
        model.classifier.weight.requires_grad = True
        model.classifier.bias.requires_grad = True
    else:
        print("Full ViT fine-tuning enabled.")
        for param in model.parameters():
            param.requires_grad = True

    model.to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    print(f"Trainable parameters: {sum(param.numel() for param in trainable_params):,}")

    criterion, class_weights = build_weighted_criterion(
        train_df["label"].to_numpy(),
        device,
    )
    run_metadata["loss"]["class_weights"] = class_weights
    save_run_metadata(run_metadata)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_val_pr_auc = float("-inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_seen = 0

        for batch_index, batch in enumerate(train_loader):
            if args.max_steps > 0 and batch_index >= args.max_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch["labels"])
            train_correct += (
                outputs.logits.argmax(dim=1) == batch["labels"]
            ).sum().item()
            train_seen += len(batch["labels"])

            if (batch_index + 1) % 100 == 0:
                print(f"  Train batch {batch_index + 1}/{len(train_loader)}")

        train_loss /= max(train_seen, 1)
        train_accuracy = train_correct / max(train_seen, 1)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        val_labels = []
        val_predictions = []
        val_melanoma_probabilities = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = criterion(outputs.logits, batch["labels"])
                probabilities = torch.softmax(outputs.logits, dim=1)
                predictions = outputs.logits.argmax(dim=1)

                val_loss += loss.item() * len(batch["labels"])
                val_seen += len(batch["labels"])
                val_labels.extend(batch["labels"].cpu().tolist())
                val_predictions.extend(predictions.cpu().tolist())
                val_melanoma_probabilities.extend(
                    probabilities[:, 1].cpu().tolist()
                )

        val_loss /= max(val_seen, 1)
        validation_metrics = compute_validation_metrics(
            val_labels,
            val_predictions,
            val_melanoma_probabilities,
        )
        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Validation loss: {val_loss:.4f} | "
            f"Validation PR-AUC: {validation_metrics['pr_auc']:.6f}"
        )

        checkpoint_sha256 = None
        if is_better_checkpoint(
            validation_metrics["pr_auc"],
            best_val_pr_auc,
        ):
            if sha256_file(LIVE_CHECKPOINT) != provenance[
                "live_checkpoint_sha256_before_training"
            ]:
                raise RuntimeError(
                    "Live application checkpoint changed during training."
                )
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            checkpoint_path = OUTPUT_DIR / "model.safetensors"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Saved candidate checkpoint not found: {checkpoint_path}"
                )
            checkpoint_sha256 = sha256_file(checkpoint_path)
            best_val_pr_auc = validation_metrics["pr_auc"]
            run_metadata["best_checkpoint"] = {
                "epoch": epoch + 1,
                "validation_pr_auc": best_val_pr_auc,
                "checkpoint_sha256": checkpoint_sha256,
            }
            print(f"Saved candidate checkpoint to {OUTPUT_DIR}")

        if sha256_file(LIVE_CHECKPOINT) != provenance[
            "live_checkpoint_sha256_before_training"
        ]:
            raise RuntimeError(
                "Live application checkpoint changed during training."
            )

        run_metadata["history"].append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "validation_loss": float(val_loss),
                "validation_metrics": validation_metrics,
                "candidate_checkpoint_sha256": checkpoint_sha256,
                "best_validation_pr_auc": float(best_val_pr_auc),
            }
        )
        save_run_metadata(run_metadata)

    if run_metadata["best_checkpoint"] is None:
        raise RuntimeError(
            "No checkpoint was selected by validation PR-AUC."
        )

    print("\nTraining complete.")
    print(f"Best validation PR-AUC: {best_val_pr_auc:.6f}")
    print(f"Candidate checkpoint: {OUTPUT_DIR}")
    print(f"Run metadata: {RUN_METADATA_PATH}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help=(
            "Optional batch limit. "
            "0 means full epoch."
        ),
    )

    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())