from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts import evaluate_vit, train_classifier_fixed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "datasets" / "Ham10000" / "processed"


@pytest.mark.parametrize(
    ("filename", "split_name"),
    [
        ("train.csv", "train"),
        ("validation.csv", "validation"),
    ],
)
def test_trainer_loads_fixed_manifests_and_authoritative_image_paths(
    filename,
    split_name,
):
    frame = train_classifier_fixed.load_split(
        MANIFEST_DIR / filename,
        expected_split=split_name,
    )

    assert frame["split"].eq(split_name).all()
    assert {"image_id", "image_path", "lesion_id", "label", "label_name"} <= set(frame)
    assert set(frame["label"]) == {0, 1}
    assert all(
        train_classifier_fixed.resolve_image_path(path).is_file()
        for path in frame["image_path"]
    )


def test_trainer_dataset_opens_image_path_not_reconstructed_image_id(tmp_path):
    image_path = tmp_path / "authoritative.jpg"
    Image.new("RGB", (9, 7), color=(10, 20, 30)).save(image_path)
    frame = pd.DataFrame(
        [
            {
                "image_id": "ID_THAT_DOES_NOT_MATCH_FILE",
                "image_path": str(image_path),
                "lesion_id": "lesion-1",
                "label": 1,
                "label_name": "mel",
            }
        ]
    )

    image, label = train_classifier_fixed.HAMDataset(
        frame,
        augment=False,
    )[0]

    assert image.size == (9, 7)
    assert label == 1


def test_evaluator_loads_held_out_test_manifest_and_keeps_lesion_provenance():
    frame = evaluate_vit.load_test_split(MANIFEST_DIR / "test.csv")

    assert frame["split"].eq("test").all()
    assert {"image_id", "image_path", "lesion_id", "label", "label_name"} <= set(frame)
    assert set(frame["label"]) == {0, 1}
    assert frame["lesion_id"].notna().all()
    assert all(
        evaluate_vit.resolve_image_path(path).is_file()
        for path in frame["image_path"]
    )


def test_evaluator_rejects_non_test_manifest(tmp_path):
    wrong_split = tmp_path / "validation.csv"
    pd.DataFrame(
        [
            {
                "split": "validation",
                "image_id": "ISIC_1",
                "image_path": "data/HAM10000/images/ISIC_1.jpg",
                "lesion_id": "lesion-1",
                "label": 0,
                "label_name": "non_mel",
            }
        ]
    ).to_csv(wrong_split, index=False)

    with pytest.raises(ValueError, match="split='test'"):
        evaluate_vit.load_test_split(wrong_split)


def test_trainer_rejects_label_name_mismatch(tmp_path):
    invalid_manifest = tmp_path / "invalid.csv"
    pd.DataFrame(
        [
            {
                "split": "train",
                "image_id": "ISIC_1",
                "image_path": "data/HAM10000/images/ISIC_1.jpg",
                "lesion_id": "lesion-1",
                "label": 1,
                "label_name": "non_mel",
            },
            {
                "split": "train",
                "image_id": "ISIC_2",
                "image_path": "data/HAM10000/images/ISIC_2.jpg",
                "lesion_id": "lesion-2",
                "label": 0,
                "label_name": "mel",
            },
        ]
    ).to_csv(invalid_manifest, index=False)

    with pytest.raises(ValueError, match="label_name"):
        train_classifier_fixed.load_split(
            invalid_manifest,
            expected_split="train",
        )
