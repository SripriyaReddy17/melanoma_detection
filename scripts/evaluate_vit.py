from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import ViTForImageClassification, ViTImageProcessor


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_CSV = BASE_DIR / "datasets" / "Ham10000" / "processed" / "test.csv"
DEFAULT_MODEL_DIR = (
    BASE_DIR / "models" / "classification" / "vit_candidate_run_001"
)

MANIFEST_COLUMNS = {
    "split",
    "image_id",
    "image_path",
    "lesion_id",
    "label",
    "label_name",
}

OUTPUT_DIR = BASE_DIR / "models" / "classification" / "evaluation"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a ViT checkpoint on the fixed HAM10000 test split."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Explicit ViT checkpoint directory. "
            f"Defaults to the candidate run: {DEFAULT_MODEL_DIR}"
        ),
    )
    return parser


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
    with path.open("rb") as file_obj:
        return hashlib.file_digest(file_obj, "sha256").hexdigest()


def load_test_split(csv_path: Path = TEST_CSV) -> pd.DataFrame:
    """Load and validate the fixed test manifest without resplitting."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Test split file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing)}"
        )
    if df.empty:
        raise ValueError(f"{csv_path.name} contains no manifest rows.")
    if df["split"].isna().any() or not df["split"].eq("test").all():
        raise ValueError(f"{csv_path.name} must contain only split='test' rows.")
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
    if set(df["label"]) != {0, 1}:
        raise ValueError(f"{csv_path.name} must include both binary classes 0 and 1.")

    df["image_path"] = df["image_path"].astype(str)
    for image_path in df["image_path"]:
        resolve_image_path(image_path)
    return df.reset_index(drop=True)


def main(model_dir: Path | None = None):
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR).resolve()

    print("=" * 70)
    print("CEFM ViT — Full HAM10000 Test Evaluation")
    print("=" * 70)
    print(f"Selected checkpoint: {model_dir}")

    if not model_dir.is_dir():
        raise FileNotFoundError(f"ViT model directory not found: {model_dir}")

    # ------------------------------------------------------------
    # Load test set
    # ------------------------------------------------------------
    df = load_test_split()

    print(f"\nTest samples: {len(df)}")
    print(f"Non-melanoma: {(df['label'] == 0).sum()}")
    print(f"Melanoma:     {(df['label'] == 1).sum()}")

    # ------------------------------------------------------------
    # Load ViT
    # ------------------------------------------------------------
    print("\nLoading ViT model...")

    processor = ViTImageProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    model = ViTForImageClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device: {device}")

    # ------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------
    y_true = []
    y_pred = []
    y_prob = []

    print("\nRunning inference...")

    with torch.no_grad():

        for i, row in df.iterrows():

            image_path = resolve_image_path(str(row["image_path"]))

            image = Image.open(image_path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )[0]

            prediction = int(torch.argmax(probabilities).item())

            # Class 1 = melanoma
            melanoma_probability = float(probabilities[1].item())

            y_true.append(int(row["label"]))
            y_pred.append(prediction)
            y_prob.append(melanoma_probability)

            if (i + 1) % 100 == 0 or i + 1 == len(df):
                print(f"Processed {i + 1}/{len(df)}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
        )
    except ValueError:
        roc_auc = None

    # ------------------------------------------------------------
    # Confusion matrix
    #
    # [[TN, FP],
    #  [FN, TP]]
    # ------------------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    sensitivity = recall

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------
    results = {
        "model": "ViT-Melanoma",
        "backbone": "google/vit-base-patch16-224",
        "checkpoint_directory": str(model_dir),
        "checkpoint_sha256": (
            sha256_file(model_dir / "model.safetensors")
            if (model_dir / "model.safetensors").is_file()
            else None
        ),
        "dataset": "HAM10000",
        "test_samples": int(len(y_true)),
        "non_melanoma_samples": int((y_true == 0).sum()),
        "melanoma_samples": int((y_true == 1).sum()),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Balanced Accuracy:  {balanced_accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Sensitivity/Recall: {sensitivity:.4f}")
    print(f"Specificity:        {specificity:.4f}")
    print(f"F1 Score:           {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:            {roc_auc:.4f}")
    else:
        print("ROC-AUC:            unavailable")

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 NonMel   Mel")
    print(f"Actual NonMel    {tn:6d}  {fp:5d}")
    print(f"Actual Mel       {fn:6d}  {tp:5d}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["non_mel", "mel"],
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "vit_test_results.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------
    prediction_df = df.copy()

    prediction_df["predicted_label"] = y_pred
    prediction_df["melanoma_probability"] = y_prob

    predictions_path = OUTPUT_DIR / "vit_test_predictions.csv"

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    print("\nSaved:")
    print(results_path)
    print(predictions_path)


if __name__ == "__main__":
    main(build_arg_parser().parse_args().model_dir)