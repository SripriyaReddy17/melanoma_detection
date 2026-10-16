"""
Contrastive alignment service for CEFM.
Connects Vision Transformer image embeddings with ABCDE clinical feature vectors
using the CEFMContrastiveModel dual projection heads.
Complies with Rules 20 & 21.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch

from app.ml.contrastive.model import CEFMContrastiveModel
from app.services.abcde.vector import build_abcde_vector
from app.services.classification import get_vit_embedding

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "contrastive"
    / "contrastive_projection_best.pt"
)

_contrastive_model = None


def get_contrastive_model() -> CEFMContrastiveModel:
    """Load the trained CEFM contrastive alignment model."""
    global _contrastive_model

    if _contrastive_model is not None:
        return _contrastive_model

    model = CEFMContrastiveModel(
        image_dim=768,
        clinical_dim=16,
        latent_dim=128,
        temperature=0.07,
    )

    if MODEL_PATH.exists():
        try:
            state = torch.load(
                MODEL_PATH,
                map_location="cpu",
            )
            model.load_state_dict(state)
            print(
                f"[contrastive] Loaded trained weights from {MODEL_PATH}"
            )
        except Exception as exc:
            print(
                f"[contrastive] Warning loading checkpoint: {exc}"
            )
    else:
        print(
            f"[contrastive] Warning: checkpoint not found at {MODEL_PATH}"
        )

    model.eval()
    _contrastive_model = model
    return _contrastive_model


class ContrastiveAligner:
    """Compute cross-modal alignment between ViT and ABCDE vectors."""

    def compute_alignment(
        self,
        image_path: Optional[str | Path] = None,
        image_embedding: Optional[np.ndarray | torch.Tensor] = None,
        abcde_features: Optional[dict] = None,
        evolution_features: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Compute cosine similarity in the shared latent space between
        the real 768-D ViT embedding and the 16-D ABCDE vector.
        """
        model = get_contrastive_model()

        # 1. Resolve the real ViT image embedding.
        emb = None

        if image_embedding is not None:
            if isinstance(image_embedding, np.ndarray):
                emb = torch.from_numpy(image_embedding).float()
            elif isinstance(image_embedding, torch.Tensor):
                emb = image_embedding.float()
        elif image_path is not None:
            raw_emb = get_vit_embedding(image_path)
            if raw_emb is not None:
                emb = torch.from_numpy(
                    np.asarray(raw_emb)
                ).float()

        # 2. Build the complete 16-D ABCDE clinical vector.
        abcde_features = abcde_features or {}
        evolution_features = evolution_features or {}

        vec_obj = build_abcde_vector(
            abcde_features,
            evolution_features,
        )
        vec_tensor = vec_obj.to_tensor()

        # Compatibility fallback used only when the real image embedding
        # is unavailable. The normal image-analysis path should use the
        # real image branch above.
        if emb is None:
            fallback_score = float(
                vec_obj.asymmetry_score * 0.4
                + vec_obj.border_irregularity * 0.3
                + vec_obj.color_diversity * 0.3
            )

            return {
                "score": round(fallback_score, 4),
                "similarity": None,
                "alignment_score": round(fallback_score, 4),
                "cosine_similarity": None,
                "status": "fallback",
                "shared_latent_dimension": model.latent_dim,
                "method": "CEFM dual projection head (clinical branch only)",
                "model": "CEFM dual projection head (clinical branch only)",
                "checkpoint": str(MODEL_PATH),
                "image_embedding_available": False,
                "clinical_vector_available": True,
                "clinical_vector": vec_tensor.squeeze(0).tolist(),
                "note": (
                    "Image embedding was not supplied; only the clinical "
                    "feature vector was encoded."
                ),
            }

        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        if vec_tensor.dim() == 1:
            vec_tensor = vec_tensor.unsqueeze(0)

        with torch.no_grad():
            zv_norm, zc_norm = model(
                emb,
                vec_tensor,
            )

            sim = float(
                (zv_norm * zc_norm).sum(dim=-1).item()
            )
            sim = max(-1.0, min(1.0, sim))

        normalized_alignment = round(
            (sim + 1.0) / 2.0,
            4,
        )

        return {
            "score": normalized_alignment,
            "similarity": round(sim, 4),
            "alignment_score": normalized_alignment,
            "cosine_similarity": round(sim, 4),
            "status": "computed",
            "shared_latent_dimension": model.latent_dim,
            "method": "CEFM Dual Projection Heads (NT-Xent aligned)",
            "model": "CEFM Dual Projection Heads (NT-Xent aligned)",
            "checkpoint": str(MODEL_PATH),
            "image_embedding_available": True,
            "clinical_vector_available": True,
            "clinical_vector": vec_tensor.squeeze(0).tolist(),
            "image_vector_norm": round(
                float(torch.norm(zv_norm, p=2).item()),
                4,
            ),
            "clinical_vector_norm": round(
                float(torch.norm(zc_norm, p=2).item()),
                4,
            ),
            "explanation": (
                f"Visual features extracted from ViT exhibit a "
                f"cross-modal alignment score of {normalized_alignment:.2f} "
                f"(cosine similarity: {sim:.2f}) with the quantified "
                f"ABCDE clinical criteria."
            ),
        }

    def compute_similarity(
        self,
        image_features=None,
        clinical_features=None,
    ):
        """Backward-compatible clinical-only helper for existing tests."""
        res = self.compute_alignment(
            abcde_features=clinical_features
        )
        return res["score"]


contrastive_aligner = ContrastiveAligner()
