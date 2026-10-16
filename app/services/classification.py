"""
Melanoma classification service using Vision Transformer (ViT).
Implements binary classification:
  Class 0 = non_mel (Benign / non-melanoma)
  Class 1 = mel (Melanoma)
Complies with Rules 2, 18, 35, 42.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import cv2
import numpy as np
import torch

from app.core.config import CLASSIFICATION_MODEL_DIR

MODEL_BACKBONE = "google/vit-base-patch16-224"
MODEL_PATH = CLASSIFICATION_MODEL_DIR / "vit_model_experiment"
MODEL_NAME = "ViT-Melanoma"
MODEL_VERSION = "v2.0"
DATASET_NAME = "HAM10000"

_classifier = None
_processor = None
_loaded_path = None


def _load_torch_model():
    """Load the fine-tuned binary ViT model and image processor."""
    global _classifier, _loaded_path, _processor

    if not MODEL_PATH.exists():
        _loaded_path = "missing"
        return None

    try:
        from transformers import ViTForImageClassification, ViTImageProcessor

        model = ViTForImageClassification.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )
        processor = ViTImageProcessor.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )
        model.eval()

        _classifier = model
        _processor = processor
        _loaded_path = str(MODEL_PATH)
        return model

    except Exception as exc:
        print(f"[classification] Could not load ViT model: {exc}")
        _loaded_path = "error"
        return None


def get_vit_embedding(image_path: str | Path) -> Optional[np.ndarray]:
    """
    Extract 768-dimensional ViT CLS embedding from penultimate layer.
    Used for cross-modal contrastive alignment.
    """
    global _classifier, _processor
    if _classifier is None:
        _load_torch_model()
    if _classifier is None or _processor is None:
        return None

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = _processor(images=rgb, return_tensors="pt")

        with torch.no_grad():
            if hasattr(_classifier, "vit"):
                outputs = _classifier.vit(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            else:
                outputs = _classifier(**inputs, output_hidden_states=True)
                cls_embedding = outputs.hidden_states[-1][:, 0, :]

        return cls_embedding.cpu().numpy().squeeze(0)
    except Exception as exc:
        print(f"[classification] Error extracting ViT embedding: {exc}")
        return None
def _binary_melanoma_probability(probabilities: dict) -> float:
    """Return the probability associated with the melanoma class."""
    if not probabilities:
        return 0.0

    if "mel" in probabilities:
        return float(probabilities["mel"])

    if "melanoma" in probabilities:
        return float(probabilities["melanoma"])

    # Fallback for binary class ordering: index 1 = melanoma.
    values = list(probabilities.values())
    if len(values) >= 2:
        return float(values[1])

    return float(values[0])


def _heuristic_risk(features: dict) -> float:
    """
    Compute a simple feature-based risk score for testing/auxiliary analysis.

    This does NOT override the ViT prediction.
    """
    score = 0.0

    asymmetry = float(features.get("asymmetry", 0.0) or 0.0)
    border = float(features.get("border_irregularity", 0.0) or 0.0)
    color = float(features.get("color_index", 0.0) or 0.0)
    diameter = float(features.get("diameter_mm", 0.0) or 0.0)
    evolution = bool(features.get("evolution_change_detected", False))

    if asymmetry >= 0.10:
        score += 0.20

    if border >= 0.50:
        score += 0.20

    if color >= 0.25:
        score += 0.20

    if diameter >= 20.0:
        score += 0.20

    if evolution:
        score += 0.20

    return min(score, 1.0)

def _risk_level(mel_prob: float) -> str:
    """Categorize probability into risk band."""
    if mel_prob >= 0.60:
        return "high"
    if mel_prob >= 0.30:
        return "medium"
    return "low"


def classify(
    image_path: str | Path,
    features: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Classify the lesion using the fine-tuned Vision Transformer.
    Complies with Rule 35: The ML model's prediction is NOT arbitrarily overridden by heuristics.
    """
    global _classifier, _processor

    if _classifier is None and _loaded_path is None:
        _load_torch_model()

    if _classifier is None:
        return {
            "method": "Not available",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "dataset": DATASET_NAME,
            "status": "Not trained / Not available",
            "predicted_class": "Not available",
            "probabilities": None,
            "melanoma_probability": None,
            "risk_level": "Not available",
            "evaluation_status": "Not evaluated",
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Unable to read image at {image_path}")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = _processor(images=rgb, return_tensors="pt")

        with torch.no_grad():
            outputs = _classifier(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        id2label = getattr(_classifier.config, "id2label", {0: "non_mel", 1: "mel"})
        # Map label names
        label_names = [id2label.get(i, str(i)) for i in range(len(probs))]
        prob_map = {name: round(float(p), 4) for name, p in zip(label_names, probs)}

        # Melanoma probability
        if "mel" in prob_map:
            mel_prob = prob_map["mel"]
        elif len(probs) >= 2:
            mel_prob = round(float(probs[1]), 4)
        else:
            mel_prob = round(float(probs[0]), 4)

        predicted_class = "mel" if mel_prob >= 0.50 else "non_mel"

        return {
            "method": "trained_vit_model",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "backbone": MODEL_BACKBONE,
            "dataset": DATASET_NAME,
            "predicted_class": predicted_class,
            "probabilities": prob_map,
            "melanoma_probability": mel_prob,
            "risk_level": _risk_level(mel_prob),
            "evaluation_status": "Model loaded and verified",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as exc:
        print(f"[classification] ViT inference error: {exc}")
        return {
            "method": "trained_vit_model",
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "dataset": DATASET_NAME,
            "status": f"Error: {exc}",
            "predicted_class": "Not available",
            "probabilities": None,
            "melanoma_probability": None,
            "risk_level": "Not available",
            "evaluation_status": "Execution error",
            "timestamp": datetime.utcnow().isoformat(),
        }


classifier = type(
    "ClassifierService",
    (),
    {
        "predict": staticmethod(classify),
        "extract_embedding": staticmethod(get_vit_embedding),
    },
)()
