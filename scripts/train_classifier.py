"""
LEGACY training workflow; not the controlled fixed-split candidate workflow.
This script writes the old ``models/classification/vit_model`` path and must
not be used for the live serving model.

Usage:
    python scripts/train_classifier.py --epochs 5 --batch-size 8

Use ``scripts/train_classifier_fixed.py`` for the isolated candidate workflow.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

IMAGE_SIZE = 224
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MODEL_BACKBONE = "google/vit-base-patch16-224"
TARGET_ACCURACY = 0.40
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-5


def find_dataset():
    """Locate HAM10000 images + metadata."""
    candidates = [
        BASE_DIR / "HAM10000",
        BASE_DIR / "data" / "HAM10000",
        BASE_DIR.parent / "HAM10000",
        BASE_DIR.parent.parent / "HAM10000",
    ]
    for root in candidates:
        if not root.exists():
            continue
        meta = root / "HAM10000_metadata.csv"
        if meta.exists():
            img_dirs = [
                d for d in root.iterdir()
                if d.is_dir() and "image" in d.name.lower()
            ]
            if img_dirs:
                return meta, img_dirs
    raise FileNotFoundError(
        "Could not locate HAM10000 metadata. Expected one of: "
        "HAM10000/HAM10000_metadata.csv, data/HAM10000/HAM10000_metadata.csv"
    )


def load_metadata():
    meta_path, img_dirs = find_dataset()
    df = pd.read_csv(meta_path)

    lookup = {}
    for d in img_dirs:
        for p in d.glob("*.jpg"):
            lookup[p.stem] = p

    df = df[df["image_id"].isin(lookup.keys())].copy()
    df["path"] = df["image_id"].map(lookup)
    df["label"] = df["dx"].map({c: i for i, c in enumerate(HAM_CLASSES)})
    df = df.dropna(subset=["label"])
    print(f"Found {len(df)} images across {len(img_dirs)} directories")
    print(df["dx"].value_counts())
    return df[["path", "label", "dx"]].reset_index(drop=True)


def build_model(num_classes=len(HAM_CLASSES), class_names=None):
    from transformers import ViTForImageClassification

    class_names = class_names or HAM_CLASSES
    label2id = {label: idx for idx, label in enumerate(class_names)}
    id2label = {idx: label for idx, label in enumerate(class_names)}

    model = ViTForImageClassification.from_pretrained(MODEL_BACKBONE)
    model.config.num_labels = num_classes
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.num_labels = num_classes
    model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)
    model.config.problem_type = "single_label_classification"
    return model


def set_trainable_parameters(model, freeze_backbone: bool = True):
    """Freeze the pretrained ViT backbone to speed up training with a lightweight head-only fine-tune."""
    for param in model.parameters():
        param.requires_grad = True

    if freeze_backbone:
        if hasattr(model, "vit"):
            for param in model.vit.parameters():
                param.requires_grad = False
        elif hasattr(model, "backbone"):
            for param in model.backbone.parameters():
                param.requires_grad = False

    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    return model


def stratified_split(frame, val_split=0.15, seed=42):
    """Split the dataset while preserving class balance."""
    rng = np.random.RandomState(seed)
    by_label = {}
    for idx, row in frame.iterrows():
        by_label.setdefault(int(row["label"]), []).append(idx)

    val_idx = []
    for label, indices in by_label.items():
        if len(indices) == 1:
            val_idx.append(indices[0])
            continue
        n_val = max(1, int(round(len(indices) * val_split)))
        n_val = min(n_val, len(indices) - 1)
        rng.shuffle(indices)
        val_idx.extend(indices[:n_val])

    val_idx = sorted(set(val_idx))
    train_idx = [idx for idx in range(len(frame)) if idx not in set(val_idx)]
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[val_idx].reset_index(drop=True)


def make_balanced_sampler(labels, num_classes):
    """Sample minibatches with near-uniform class exposure to reduce majority-class domination."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.max(counts) / np.clip(counts, 1.0, None)
    sample_weights = weights[labels]
    from torch.utils.data import WeightedRandomSampler

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = np.max(counts) / counts
    return torch.tensor(weights, dtype=torch.float32)


def train(args):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import ViTImageProcessor

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    df = load_metadata()
    if args.binary_melanoma:
        df["dx"] = df["dx"].astype(str)
        df["label"] = df["dx"].eq("mel").astype(int)
        class_names = ["non_mel", "mel"]
        print("Training binary melanoma-vs-rest classifier.")
    else:
        class_names = HAM_CLASSES

    if args.max_samples and len(df) > args.max_samples:
        df = df.iloc[: args.max_samples].reset_index(drop=True)
        print(f"Using a reduced sample set of {len(df)} items for quick validation")
    processor = ViTImageProcessor.from_pretrained(MODEL_BACKBONE)

    class HAMDataset(Dataset):
        def __init__(self, frame, augment=True):
            self.frame = frame
            self.augment = augment

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            row = self.frame.iloc[idx]
            img = Image.open(row["path"]).convert("RGB")

            if self.augment:
                if np.random.rand() < 0.5:
                    img = ImageOps.mirror(img)
                if np.random.rand() < 0.5:
                    img = img.rotate(np.random.uniform(-20, 20), resample=Image.BILINEAR)
                if np.random.rand() < 0.25:
                    img = ImageOps.autocontrast(img)

            return img, int(row["label"])

    def collate_fn(batch):
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    train_df, val_df = stratified_split(df, val_split=args.val_split, seed=42)
    print(f"Train set size: {len(train_df)} | Validation set size: {len(val_df)}")

    train_sampler = make_balanced_sampler(train_df["label"].to_numpy(), len(class_names)) if args.balanced_sampling else None
    train_loader = DataLoader(
        HAMDataset(train_df),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        HAMDataset(val_df, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = build_model(num_classes=len(class_names), class_names=class_names).to(device)
    set_trainable_parameters(model, freeze_backbone=args.freeze_backbone)
    if args.freeze_backbone:
        print("Stage 1: freezing the ViT backbone and training only the classifier head.")
    else:
        print("Full fine-tuning enabled: all ViT parameters are trainable.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    class_weights = compute_class_weights(train_df["label"].to_numpy(), len(class_names)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = BASE_DIR / "models" / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dir = out_dir / "vit_model"
    best_acc = 0.0
    patience_counter = 0

    if args.freeze_backbone and args.unfreeze_last_blocks > 0:
        stage_switch_epoch = max(1, args.epochs - max(1, args.unfreeze_last_blocks))
    else:
        stage_switch_epoch = args.epochs

    for epoch in range(args.epochs):
        if args.freeze_backbone and args.unfreeze_last_blocks > 0 and epoch == stage_switch_epoch - 1:
            if hasattr(model, "vit") and hasattr(model.vit, "encoder") and hasattr(model.vit.encoder, "layer"):
                layers = list(model.vit.encoder.layer)
                for block in layers[-args.unfreeze_last_blocks :]:
                    for param in block.parameters():
                        param.requires_grad = True
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable_params, lr=max(args.lr * 0.1, 1e-6), weight_decay=1e-4)
                print(f"Stage 2: unfreezing the last {args.unfreeze_last_blocks} ViT blocks and lowering the learning rate for refinement.")
                print(f"Trainable parameters now: {sum(p.numel() for p in trainable_params):,}")

        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        train_batches = 0
        for batch in train_loader:
            if args.max_steps and train_batches >= args.max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits
            loss = criterion(logits, batch["labels"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch["labels"])
            correct += (logits.argmax(1) == batch["labels"]).sum().item()
            seen += len(batch["labels"])
            train_batches += 1

        scheduler.step()
        train_acc = correct / max(seen, 1)

        model.eval()
        v_correct, v_seen = 0, 0
        with torch.no_grad():
            val_batches = 0
            for batch in val_loader:
                if args.max_steps and val_batches >= args.max_steps:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                v_correct += (outputs.logits.argmax(1) == batch["labels"]).sum().item()
                v_seen += len(batch["labels"])
                val_batches += 1
        val_acc = v_correct / max(v_seen, 1)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"loss={total_loss/max(seen,1):.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  saved new best -> {save_dir}")
        else:
            patience_counter += 1

        if args.early_stopping and patience_counter >= 3:
            print("Early stopping triggered: validation accuracy plateaued.")
            break

    print(f"\nDone. Best val_acc={best_acc:.4f}. Target={TARGET_ACCURACY:.2%}. Model at {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze the pretrained ViT backbone and train only the classifier head for a fast transfer-learning pass.")
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True,
                        help="Use class-balanced sampling to counter HAM10000's heavy class imbalance.")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2,
                        help="Number of final ViT encoder blocks to unfreeze after the initial head training stage.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional subset for quick validation runs; 0 means use all data.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on batches per epoch for quick validation; 0 means full epoch.")
    parser.add_argument("--binary-melanoma", action="store_true", help="Train a melanoma-vs-rest binary classifier instead of the full 7-class HAM10000 model.")
    parser.add_argument("--early-stopping", action="store_true", help="Stop if validation accuracy stops improving.")
    train(parser.parse_args())
