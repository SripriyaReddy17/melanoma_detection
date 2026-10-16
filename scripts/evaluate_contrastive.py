"""
Evaluate Cross-Modal Contrastive Learning Alignment for CEFM.

Uses real HAM10000 test-set image embeddings and ABCDE feature vectors.

Measures:
- Positive-pair cosine similarities
- Negative-pair cosine similarities
- Mean/std statistics
- Positive-vs-negative separation
- Image -> Clinical Top-1 retrieval accuracy
- Clinical -> Image Top-1 retrieval accuracy

The test set is kept untouched during contrastive training.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ml.contrastive.model import CEFMContrastiveModel
from app.services.classification import get_vit_embedding
from app.services.segmentation import segment_lesion
from app.services.analysis import extract_abcde_features
from app.services.abcde.vector import build_abcde_vector


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "contrastive"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CHECKPOINT_PATH = (
    BASE_DIR
    / "models"
    / "contrastive"
    / "contrastive_projection_best.pt"
)


METADATA_PATH = (
    BASE_DIR
    / "datasets"
    / "Ham10000"
    / "processed"
    / "test.csv"
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


# ============================================================
# Collect real test pairs
# ============================================================

def collect_real_pairs(num_pairs=1000):
    """
    Collect real HAM10000 test-set ViT + ABCDE feature pairs.
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"HAM10000 test metadata not found:\n"
            f"{METADATA_PATH}"
        )

    print(
        f"[Contrastive Evaluation] "
        f"Using metadata: {METADATA_PATH}"
    )

    df = pd.read_csv(METADATA_PATH)

    print(
        f"[Contrastive Evaluation] "
        f"Test rows available: {len(df)}"
    )

    # --------------------------------------------------------
    # Build image map
    # --------------------------------------------------------

    disk_map = {}

    for directory in IMAGE_DIRS:

        if directory.exists():

            for image_path in directory.glob("*.jpg"):

                disk_map[
                    image_path.stem
                ] = image_path

    print(
        f"[Contrastive Evaluation] "
        f"Found {len(disk_map)} HAM10000 images."
    )

    # --------------------------------------------------------
    # Collect pairs
    # --------------------------------------------------------

    pairs = []

    for _, row in df.iterrows():

        if len(pairs) >= num_pairs:
            break

        image_id = str(
            row["image_id"]
        )

        if image_id not in disk_map:
            print(
                f"[Contrastive Evaluation] "
                f"Image not found: {image_id}"
            )
            continue

        image_path = disk_map[image_id]

        try:

            print(
                f"[Contrastive Evaluation] "
                f"Processing {len(pairs) + 1}/{num_pairs}: "
                f"{image_id}"
            )

            # ------------------------------------------------
            # ViT embedding
            # ------------------------------------------------

            embedding = get_vit_embedding(
                image_path
            )

            if embedding is None:

                print(
                    f"[Contrastive Evaluation] "
                    f"Skipping {image_id}: "
                    f"ViT embedding unavailable."
                )

                continue

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

            if embedding.shape[0] != 768:

                print(
                    f"[Contrastive Evaluation] "
                    f"Skipping {image_id}: "
                    f"expected 768-D embedding, "
                    f"got {embedding.shape[0]}."
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
                    f"[Contrastive Evaluation] "
                    f"Skipping {image_id}: "
                    f"image could not be loaded."
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
                    f"[Contrastive Evaluation] "
                    f"Skipping {image_id}: "
                    f"segmentation failed."
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
                    f"[Contrastive Evaluation] "
                    f"Skipping {image_id}: "
                    f"ABCDE extraction failed."
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
                    f"[Contrastive Evaluation] "
                    f"Skipping {image_id}: "
                    f"expected 16-D ABCDE vector, "
                    f"got {clinical_vector.shape[0]}."
                )

                continue

            # ------------------------------------------------
            # Store pair
            # ------------------------------------------------

            pairs.append(
                (
                    embedding,
                    clinical_vector
                )
            )

        except Exception as e:

            print(
                f"[Contrastive Evaluation] "
                f"Skipping {image_id} due to error: {e}"
            )

    if len(pairs) < 4:

        raise RuntimeError(
            f"Only {len(pairs)} real test pairs were collected. "
            f"At least 4 pairs are required for evaluation."
        )

    print(
        f"\n[Contrastive Evaluation] "
        f"Collected {len(pairs)} REAL HAM10000 TEST pairs."
    )

    return pairs


# ============================================================
# Evaluate contrastive model
# ============================================================

def evaluate_contrastive(
    num_pairs=1000,
    device="cpu"
):

    print(
        "\n[Contrastive Evaluation] "
        "Evaluating real cross-modal alignment "
        "on the untouched HAM10000 test set..."
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        device
    )

    print(
        f"[Contrastive Evaluation] "
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = CEFMContrastiveModel(
        image_dim=768,
        clinical_dim=16,
        latent_dim=128
    ).to(device)

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Best contrastive checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint
    )

    print(
        f"[Contrastive Evaluation] "
        f"Loaded BEST checkpoint from:\n"
        f"{CHECKPOINT_PATH}"
    )

    model.eval()

    # --------------------------------------------------------
    # Collect test pairs
    # --------------------------------------------------------

    pairs = collect_real_pairs(
        num_pairs=num_pairs
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    image_embs = torch.tensor(
        np.stack(
            [
                pair[0]
                for pair in pairs
            ]
        ),
        dtype=torch.float32
    ).to(device)

    clinical_vecs = torch.tensor(
        np.stack(
            [
                pair[1]
                for pair in pairs
            ]
        ),
        dtype=torch.float32
    ).to(device)

    # --------------------------------------------------------
    # Cross-modal projection
    # --------------------------------------------------------

    with torch.no_grad():

        image_projection, clinical_projection = model(
            image_embs,
            clinical_vecs
        )

        # ----------------------------------------------------
        # Positive pairs
        # ----------------------------------------------------

        positive_similarities = (
            image_projection
            * clinical_projection
        ).sum(
            dim=-1
        )

        # ----------------------------------------------------
        # Complete similarity matrix
        # ----------------------------------------------------

        similarity_matrix = torch.matmul(
            image_projection,
            clinical_projection.T
        )

        # ----------------------------------------------------
        # Negative pairs
        # ----------------------------------------------------

        n = len(pairs)

        negative_mask = (
            ~torch.eye(
                n,
                dtype=torch.bool,
                device=device
            )
        )

        negative_similarities = (
            similarity_matrix[
                negative_mask
            ]
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    pos_sims = (
        positive_similarities
        .cpu()
        .numpy()
    )

    neg_sims = (
        negative_similarities
        .cpu()
        .numpy()
    )

    similarity_matrix_np = (
        similarity_matrix
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    pos_mean = float(
        np.mean(pos_sims)
    )

    pos_std = float(
        np.std(pos_sims)
    )

    neg_mean = float(
        np.mean(neg_sims)
    )

    neg_std = float(
        np.std(neg_sims)
    )

    separation = (
        pos_mean
        - neg_mean
    )

    # --------------------------------------------------------
    # Retrieval accuracy
    # --------------------------------------------------------

    image_to_clinical_predictions = (
        np.argmax(
            similarity_matrix_np,
            axis=1
        )
    )

    clinical_to_image_predictions = (
        np.argmax(
            similarity_matrix_np,
            axis=0
        )
    )

    correct_image_to_clinical = (
        image_to_clinical_predictions
        == np.arange(n)
    )

    correct_clinical_to_image = (
        clinical_to_image_predictions
        == np.arange(n)
    )

    image_to_clinical_accuracy = float(
        np.mean(
            correct_image_to_clinical
        )
    )

    clinical_to_image_accuracy = float(
        np.mean(
            correct_clinical_to_image
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "evaluation_type":
            "real_ham10000_test_cross_modal_alignment",

        "checkpoint":
            str(CHECKPOINT_PATH),

        "metadata":
            str(METADATA_PATH),

        "num_pairs_evaluated":
            len(pairs),

        "positive_pairs": {

            "mean_cosine_similarity":
                round(
                    pos_mean,
                    4
                ),

            "std_cosine_similarity":
                round(
                    pos_std,
                    4
                ),

            "min":
                round(
                    float(
                        np.min(
                            pos_sims
                        )
                    ),
                    4
                ),

            "max":
                round(
                    float(
                        np.max(
                            pos_sims
                        )
                    ),
                    4
                ),
        },

        "negative_pairs": {

            "mean_cosine_similarity":
                round(
                    neg_mean,
                    4
                ),

            "std_cosine_similarity":
                round(
                    neg_std,
                    4
                ),

            "min":
                round(
                    float(
                        np.min(
                            neg_sims
                        )
                    ),
                    4
                ),

            "max":
                round(
                    float(
                        np.max(
                            neg_sims
                        )
                    ),
                    4
                ),
        },

        "alignment_separation":
            round(
                float(
                    separation
                ),
                4
            ),

        "retrieval": {

            "image_to_clinical_top1_accuracy":
                round(
                    image_to_clinical_accuracy,
                    4
                ),

            "clinical_to_image_top1_accuracy":
                round(
                    clinical_to_image_accuracy,
                    4
                ),
        },

        "note": (
            "Positive pairs are matched HAM10000 "
            "test-set image embeddings and ABCDE "
            "vectors. Negative pairs are mismatched "
            "image-clinical combinations from the "
            "same test evaluation batch."
        ),
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "contrastive_evaluation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        "REAL HAM10000 TEST "
        "CONTRASTIVE EVALUATION"
    )

    print(
        "===================================="
    )

    print(
        f"Test pairs evaluated: "
        f"{len(pairs)}"
    )

    print(
        f"Positive-pair cosine similarity: "
        f"{pos_mean:.4f} ± {pos_std:.4f}"
    )

    print(
        f"Negative-pair cosine similarity: "
        f"{neg_mean:.4f} ± {neg_std:.4f}"
    )

    print(
        f"Mean Separation (Pos - Neg): "
        f"{separation:.4f}"
    )

    print(
        f"Image → Clinical Top-1 Accuracy: "
        f"{image_to_clinical_accuracy:.4f}"
    )

    print(
        f"Clinical → Image Top-1 Accuracy: "
        f"{clinical_to_image_accuracy:.4f}"
    )

    print(
        "\nCheckpoint used:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        "\nTest metadata used:"
    )

    print(
        METADATA_PATH
    )

    print(
        "\nResults written to:"
    )

    print(
        output_file
    )

    print(
        "===================================="
    )

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    evaluate_contrastive(
        num_pairs=1000,
        device="cpu"
    )