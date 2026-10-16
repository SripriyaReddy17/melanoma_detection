import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_ham10000_splits import (
    MELANOMA_DIAGNOSIS,
    NON_MELANOMA_DIAGNOSES,
    prepare_dataset_splits,
)


def _make_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_root = tmp_path / "project"
    image_dir = project_root / "data" / "HAM10000" / "images"
    image_dir.mkdir(parents=True)
    source_path = (
        project_root
        / "data"
        / "HAM10000"
        / "ham10000_metadata_2026-09-21.csv"
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    diagnoses = [MELANOMA_DIAGNOSIS, *sorted(NON_MELANOMA_DIAGNOSES)]
    for diagnosis_index, diagnosis in enumerate(diagnoses):
        for lesion_number in range(12):
            lesion_id = f"lesion-{diagnosis_index}-{lesion_number:02d}"
            image_id = f"ISIC_{diagnosis_index:02d}{lesion_number:04d}"
            (image_dir / f"{image_id}.jpg").write_bytes(
                f"distinct image bytes {image_id}".encode()
            )
            rows.append(
                {
                    "isic_id": image_id,
                    "lesion_id": lesion_id,
                    "diagnosis_1": (
                        "Malignant" if diagnosis == MELANOMA_DIAGNOSIS else "Benign"
                    ),
                    "diagnosis_2": diagnosis,
                    "diagnosis_3": (
                        "Melanoma, NOS"
                        if diagnosis == MELANOMA_DIAGNOSIS
                        else diagnosis
                    ),
                }
            )

    # Add a second image for selected lesions to verify grouping.
    for row in rows[:8]:
        extra_id = f"{row['isic_id']}_repeat"
        (image_dir / f"{extra_id}.jpg").write_bytes(
            f"distinct image bytes {extra_id}".encode()
        )
        rows.append({**row, "isic_id": extra_id})

    rows.append(
        {
            "isic_id": "ISIC_unknown",
            "lesion_id": "lesion-unknown",
            "diagnosis_1": "Indeterminate",
            "diagnosis_2": "Indeterminate epidermal proliferations",
            "diagnosis_3": None,
        }
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(source_path, index=False)
    return project_root, source_path, image_dir


def _generate(tmp_path: Path, output_name: str = "processed") -> dict:
    project_root, source_path, image_dir = _make_source(tmp_path)
    return prepare_dataset_splits(
        source_metadata=source_path,
        image_dir=image_dir,
        output_dir=project_root / output_name,
        project_root=project_root,
        seed=42,
    )


def test_generated_splits_have_no_lesion_or_image_id_overlap(tmp_path):
    _generate(tmp_path)
    output_dir = tmp_path / "project" / "processed"
    splits = {
        name: pd.read_csv(output_dir / f"{name}.csv")
        for name in ("train", "validation", "test")
    }

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        assert set(splits[left]["lesion_id"]).isdisjoint(
            set(splits[right]["lesion_id"])
        )
        assert set(splits[left]["image_id"]).isdisjoint(
            set(splits[right]["image_id"])
        )


def test_generated_labels_paths_and_class_presence_are_valid(tmp_path):
    metadata = _generate(tmp_path)
    output_dir = tmp_path / "project" / "processed"
    for split_name in ("train", "validation", "test"):
        frame = pd.read_csv(output_dir / f"{split_name}.csv")
        assert set(frame["label"]) == {0, 1}
        assert set(frame["label_name"]) == {"non_mel", "mel"}
        assert (frame["image_id"] == frame["image_path"].map(
            lambda value: Path(value).stem
        )).all()
        assert all(
            (tmp_path / "project" / path).is_file()
            for path in frame["image_path"]
        )

    assert metadata["validation_checks"]["image_paths_exist"]["passed"] is True
    assert metadata["validation_checks"]["both_classes_in_each_split"]["passed"] is True
    assert metadata["source_counts"]["rows_unknown_excluded"] == 1


def test_split_generation_is_reproducible_for_same_seed(tmp_path):
    project_root, source_path, image_dir = _make_source(tmp_path)
    first_dir = project_root / "first"
    second_dir = project_root / "second"
    for output_dir in (first_dir, second_dir):
        prepare_dataset_splits(
            source_metadata=source_path,
            image_dir=image_dir,
            output_dir=output_dir,
            project_root=project_root,
            seed=42,
        )

    for split_name in ("train", "validation", "test"):
        first = (first_dir / f"{split_name}.csv").read_bytes()
        second = (second_dir / f"{split_name}.csv").read_bytes()
        assert first == second
    metadata = (first_dir / "split_metadata.json").read_text(encoding="utf-8")
    assert hashlib.sha256((first_dir / "test.csv").read_bytes()).hexdigest() in metadata
    assert (first_dir / "split_metadata.json").exists()


def test_preparation_refuses_to_overwrite_existing_manifests(tmp_path):
    project_root, source_path, image_dir = _make_source(tmp_path)
    output_dir = project_root / "processed"
    prepare_dataset_splits(
        source_metadata=source_path,
        image_dir=image_dir,
        output_dir=output_dir,
        project_root=project_root,
    )
    with pytest.raises(FileExistsError):
        prepare_dataset_splits(
            source_metadata=source_path,
            image_dir=image_dir,
            output_dir=output_dir,
            project_root=project_root,
        )


def test_identical_image_content_across_splits_fails_before_write(tmp_path):
    project_root, source_path, image_dir = _make_source(tmp_path)
    baseline_dir = project_root / "baseline"
    prepare_dataset_splits(
        source_metadata=source_path,
        image_dir=image_dir,
        output_dir=baseline_dir,
        project_root=project_root,
        seed=42,
    )
    train_row = pd.read_csv(baseline_dir / "train.csv").iloc[0]
    validation_row = pd.read_csv(baseline_dir / "validation.csv").iloc[0]
    train_image = project_root / train_row["image_path"]
    validation_image = project_root / validation_row["image_path"]
    validation_image.write_bytes(train_image.read_bytes())

    with pytest.raises(ValueError, match="Identical image content"):
        prepare_dataset_splits(
            source_metadata=source_path,
            image_dir=image_dir,
            output_dir=project_root / "duplicate",
            project_root=project_root,
            seed=42,
        )
