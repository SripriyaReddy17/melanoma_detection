"""
Cross-Modal Contrastive Learning Architecture for CEFM (Rule 20).
Aligns ViT visual embeddings (768-d) with ABCDE clinical feature vectors (16-d)
in a shared 128-dimensional latent space using dual projection heads and NT-Xent loss.
Complies with Equations (1)-(4) from CEFM Paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ImageProjectionHead(nn.Module):
    """
    MLP projector for Vision Transformer image features (768 -> 256 -> 128).
    Includes non-linear activation and LayerNorm.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClinicalProjectionHead(nn.Module):
    """
    MLP projector for ABCDE clinical feature vector (16 -> 64 -> 128).
    Maps clinical descriptors into the shared latent space.
    """
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CEFMContrastiveModel(nn.Module):
    """
    Dual-projection head model for cross-modal alignment between
    dermoscopic visual features and ABCDE clinical features.
    """
    def __init__(
        self,
        image_dim: int = 768,
        clinical_dim: int = 16,
        latent_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.image_proj = ImageProjectionHead(image_dim, 256, latent_dim)
        self.clinical_proj = ClinicalProjectionHead(clinical_dim, 64, latent_dim)
        self.temperature = temperature
        self.latent_dim = latent_dim

    def forward(
        self,
        image_embeddings: torch.Tensor,
        clinical_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute normalized projected embeddings.
        Returns:
            z_v: L2-normalized image projections [batch_size, latent_dim]
            z_c: L2-normalized clinical projections [batch_size, latent_dim]
        """
        # Eq (1): zv = hv(v), zc = hc(u)
        zv = self.image_proj(image_embeddings)
        zc = self.clinical_proj(clinical_vectors)

        # Eq (2): L2 normalization
        zv_norm = F.normalize(zv, p=2, dim=-1)
        zc_norm = F.normalize(zc, p=2, dim=-1)

        return zv_norm, zc_norm

    def compute_similarity(
        self,
        image_embedding: torch.Tensor,
        clinical_vector: torch.Tensor,
    ) -> float:
        """
        Compute cosine similarity between single image embedding and clinical vector.
        """
        self.eval()
        with torch.no_grad():
            if image_embedding.dim() == 1:
                image_embedding = image_embedding.unsqueeze(0)
            if clinical_vector.dim() == 1:
                clinical_vector = clinical_vector.unsqueeze(0)

            zv_norm, zc_norm = self.forward(image_embedding, clinical_vector)
            sim = (zv_norm * zc_norm).sum(dim=-1).item()
            return float(max(-1.0, min(1.0, sim)))


def nt_xent_loss(
    zv: torch.Tensor,
    zc: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).
    Symmetric contrastive loss between image (zv) and clinical (zc) pairs.
    Complies with Eqs (3) & (4) from CEFM Paper:
    L_contrastive = (1 / 2N) * sum(L_(v->c) + L_(c->v))
    """
    batch_size = zv.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=zv.device, requires_grad=True)

    # Cosine similarity matrix between all images and all clinical vectors: [B, B]
    sim_matrix = torch.matmul(zv, zc.T) / temperature

    # Ground truth targets: diagonal elements (i, i) are matched pairs
    labels = torch.arange(batch_size, device=zv.device)

    # Cross entropy v -> c and c -> v
    loss_v2c = F.cross_entropy(sim_matrix, labels)
    loss_c2v = F.cross_entropy(sim_matrix.T, labels)

    loss_contrastive = 0.5 * (loss_v2c + loss_c2v)
    return loss_contrastive
