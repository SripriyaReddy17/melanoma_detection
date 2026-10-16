"""Create auditable, lesion-level binary HAM10000 split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_METADATA = (
    BASE_DIR
    / "data"
    / "HAM10000"
    / "ham10000_metadata_2026-09-21.csv"
)
IMAGE_DIR = BASE_DIR / "data" / "HAM10000" / "images"
OUTPUT_DIR = BASE_DIR / "datasets" / "Ham10000" / "processed"

DIAGNOSIS_COLUMN = "diagnosis_2"
MELANOMA_DIAGNOSIS = "Malignant melanocytic proliferations (Melanoma)"
INDETERMINATE_DIAGNOSIS = "Indeterminate epidermal proliferations"
NON_MELANOMA_DIAGNOSES = {
    "Benign melanocytic proliferations",
    "Benign epidermal proliferations",
    "Malignant adnexal epithelial proliferations - Follicular",
    "Malignant epidermal proliferations",
    "Benign soft tissue proliferations - Vascular",
    "Benign soft tissue proliferations - Fibro-histiocytic",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    with path.open("rb") as file_obj:
        return hashlib.file_digest(file_obj, "sha256").hexdigest()


def binary_label(diagnosis: object) -> int | None:
    """Map explicitly known diagnosis_2 values; return None for unknowns."""
    if pd.isna(diagnosis):
        return None

    value = str(diagnosis).strip()
    if not value:
        return None
    if value == MELANOMA_DIAGNOSIS:
        return 1
    if value in NON_MELANOMA_DIAGNOSES:
        return 0
    return None


def _image_lookup(image_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in lookup:
            duplicates.append(path.stem)
        else:
            lookup[path.stem] = path.resolve()

    if duplicates:
        raise ValueError(
            "Image directory contains duplicate image IDs across files: "
            + ", ".join(sorted(set(duplicates))[:10])
        )
    return lookup


def _split_lesions(
    lesion_table: pd.DataFrame,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, set[str]]:
    if set(ratios) != {"train", "validation", "test"}:
        raise ValueError("Ratios must define train, validation, and test.")
    if any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError("Every split ratio must be greater than zero.")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.")

    lesion_table = lesion_table.sort_values("lesion_id").reset_index(drop=True)
    labels = lesion_table["label"]
    test_size = ratios["test"]
    validation_fraction_of_remaining = (
        ratios["validation"] / (ratios["train"] + ratios["validation"])
    )

    train_validation, test = train_test_split(
        lesion_table,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction_of_remaining,
        random_state=seed,
        stratify=train_validation["label"],
    )

    return {
        "train": set(train["lesion_id"].astype(str)),
        "validation": set(validation["lesion_id"].astype(str)),
        "test": set(test["lesion_id"].astype(str)),
    }


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "non_mel": int((frame["label"] == 0).sum()),
        "mel": int((frame["label"] == 1).sum()),
    }


def prepare_dataset_splits(
    source_metadata: Path = SOURCE_METADATA,
    image_dir: Path = IMAGE_DIR,
    output_dir: Path = OUTPUT_DIR,
    project_root: Path = BASE_DIR,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build, validate, and write train/validation/test CSVs and provenance."""
    source_metadata = Path(source_metadata).resolve()
    image_dir = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()
    project_root = Path(project_root).resolve()
    ratios = dict(ratios or DEFAULT_RATIOS)

    if not source_metadata.is_file():
        raise FileNotFoundError(f"Source metadata not found: {source_metadata}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    expected_outputs = [
        output_dir / "train.csv",
        output_dir / "validation.csv",
        output_dir / "test.csv",
        output_dir / "split_metadata.json",
    ]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "Split outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    source = pd.read_csv(source_metadata, dtype={"isic_id": str, "lesion_id": str})
    required_columns = {"isic_id", "lesion_id", DIAGNOSIS_COLUMN}
    missing_columns = required_columns - set(source.columns)
    if missing_columns:
        raise ValueError(
            f"Source metadata is missing required columns: {sorted(missing_columns)}"
        )

    if source["isic_id"].isna().any() or source["isic_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Source metadata contains missing/blank image IDs.")
    if source["lesion_id"].isna().any() or source["lesion_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Source metadata contains missing/blank lesion IDs.")
    duplicate_image_rows = int(source["isic_id"].duplicated(keep=False).sum())
    if duplicate_image_rows:
        raise ValueError(
            f"Source metadata contains {duplicate_image_rows} rows with duplicate image IDs."
        )

    source["isic_id"] = source["isic_id"].astype(str).str.strip()
    source["lesion_id"] = source["lesion_id"].astype(str).str.strip()
    source["label"] = source[DIAGNOSIS_COLUMN].map(binary_label)

    label_rows = source[source["label"].notna()].copy()
    unknown_rows = source[source["label"].isna()].copy()
    label_rows["label"] = label_rows["label"].astype(int)

    lesion_label_counts = label_rows.groupby("lesion_id")["label"].nunique()
    inconsistent_lesions = lesion_label_counts[lesion_label_counts != 1]
    if not inconsistent_lesions.empty:
        raise ValueError(
            "Diagnosis-derived binary labels conflict within lesion IDs: "
            + ", ".join(inconsistent_lesions.index.astype(str).tolist()[:10])
        )

    image_lookup = _image_lookup(image_dir)
    missing_ids = sorted(set(label_rows["isic_id"]) - set(image_lookup))
    if missing_ids:
        raise FileNotFoundError(
            f"{len(missing_ids)} labeled metadata images are missing; examples: "
            + ", ".join(missing_ids[:10])
        )

    label_rows["image_path"] = label_rows["isic_id"].map(
        lambda image_id: image_lookup[image_id].relative_to(project_root).as_posix()
    )
    label_rows["label_name"] = label_rows["label"].map(
        {0: "non_mel", 1: "mel"}
    )

    lesion_table = (
        label_rows[["lesion_id", "label"]]
        .drop_duplicates()
        .sort_values("lesion_id")
        .reset_index(drop=True)
    )
    if lesion_table["lesion_id"].duplicated().any():
        raise ValueError("A lesion was assigned more than one binary label.")

    lesion_splits = _split_lesions(lesion_table, seed, ratios)
    manifests: dict[str, pd.DataFrame] = {}
    for split_name, lesion_ids in lesion_splits.items():
        manifest = label_rows[
            label_rows["lesion_id"].isin(lesion_ids)
        ].copy()
        manifest = manifest.sort_values("isic_id").reset_index(drop=True)
        manifest.insert(0, "split", split_name)
        manifests[split_name] = manifest

    split_lesions = {
        split_name: set(frame["lesion_id"].astype(str))
        for split_name, frame in manifests.items()
    }
    split_images = {
        split_name: set(frame["isic_id"].astype(str))
        for split_name, frame in manifests.items()
    }

    lesion_overlaps = {
        "train_validation": len(split_lesions["train"] & split_lesions["validation"]),
        "train_test": len(split_lesions["train"] & split_lesions["test"]),
        "validation_test": len(
            split_lesions["validation"] & split_lesions["test"]
        ),
    }
    image_overlaps = {
        "train_validation": len(split_images["train"] & split_images["validation"]),
        "train_test": len(split_images["train"] & split_images["test"]),
        "validation_test": len(split_images["validation"] & split_images["test"]),
    }
    if any(lesion_overlaps.values()):
        raise ValueError(f"Lesion leakage detected: {lesion_overlaps}")
    if any(image_overlaps.values()):
        raise ValueError(f"Image ID leakage detected: {image_overlaps}")

    missing_manifest_paths = [
        path
        for frame in manifests.values()
        for path in frame["image_path"]
        if not (project_root / Path(path)).is_file()
    ]
    if missing_manifest_paths:
        raise FileNotFoundError(
            f"{len(missing_manifest_paths)} generated manifest image paths do not exist."
        )

    image_hash_splits: dict[str, set[str]] = {}
    for split_name, frame in manifests.items():
        image_hash_splits[split_name] = {
            sha256_file(project_root / Path(path))
            for path in frame["image_path"]
        }
    content_overlaps = {
        "train_validation": len(
            image_hash_splits["train"] & image_hash_splits["validation"]
        ),
        "train_test": len(image_hash_splits["train"] & image_hash_splits["test"]),
        "validation_test": len(
            image_hash_splits["validation"] & image_hash_splits["test"]
        ),
    }
    if any(content_overlaps.values()):
        raise ValueError(f"Identical image content crosses splits: {content_overlaps}")

    class_presence = {
        split_name: sorted(frame["label"].unique().tolist())
        for split_name, frame in manifests.items()
    }
    for split_name in ("train", "validation", "test"):
        if class_presence[split_name] != [0, 1]:
            raise ValueError(
                f"Both classes must be present in {split_name}; "
                f"observed labels={class_presence[split_name]}"
            )

    validation_checks = {
        "lesion_ids_disjoint": {
            "passed": not any(lesion_overlaps.values()),
            "overlap_counts": lesion_overlaps,
        },
        "image_ids_disjoint": {
            "passed": not any(image_overlaps.values()),
            "overlap_counts": image_overlaps,
        },
        "identical_image_content_disjoint": {
            "passed": not any(content_overlaps.values()),
            "overlap_counts": content_overlaps,
            "method": "SHA-256 comparison across all included image files",
        },
        "image_paths_exist": {
            "passed": not missing_manifest_paths,
            "missing_count": len(missing_manifest_paths),
        },
        "both_classes_in_each_split": {
            "passed": all(labels == [0, 1] for labels in class_presence.values()),
            "labels_by_split": class_presence,
        },
        "source_image_ids_unique": {
            "passed": duplicate_image_rows == 0,
            "duplicate_rows": duplicate_image_rows,
        },
        "lesion_labels_consistent": {
            "passed": inconsistent_lesions.empty,
            "inconsistent_lesion_count": int(len(inconsistent_lesions)),
        },
        "test_set_held_out": {
            "passed": True,
            "policy": (
                "This preparation script assigns the test split only; it does not "
                "train a model, tune a threshold, select a checkpoint, or inspect "
                "model performance."
            ),
        },
    }
    if not all(check["passed"] for check in validation_checks.values()):
        raise ValueError("A dataset split validation check failed.")

    output_columns = [
        "split",
        "image_id",
        "image_path",
        "lesion_id",
        "label",
        "label_name",
        DIAGNOSIS_COLUMN,
    ]
    for optional_column in ("diagnosis_1", "diagnosis_3", "diagnosis_confirm_type"):
        if optional_column in source.columns:
            output_columns.append(optional_column)
    manifests = {
        name: frame.rename(columns={"isic_id": "image_id"})[output_columns]
        for name, frame in manifests.items()
    }

    source_class_counts = {
        "non_mel": int((label_rows["label"] == 0).sum()),
        "mel": int((label_rows["label"] == 1).sum()),
    }
    source_lesion_class_counts = {
        "non_mel": int(
            lesion_table.loc[lesion_table["label"] == 0, "lesion_id"].nunique()
        ),
        "mel": int(
            lesion_table.loc[lesion_table["label"] == 1, "lesion_id"].nunique()
        ),
    }
    unknown_values = (
        unknown_rows[DIAGNOSIS_COLUMN]
        .fillna("<missing>")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    split_summary: dict[str, Any] = {}
    for split_name, frame in manifests.items():
        split_summary[split_name] = {
            "image_count": int(len(frame)),
            "lesion_count": int(frame["lesion_id"].nunique()),
            "class_counts": _class_counts(frame),
            "class_lesion_counts": {
                "non_mel": int(
                    frame.loc[frame["label"] == 0, "lesion_id"].nunique()
                ),
                "mel": int(
                    frame.loc[frame["label"] == 1, "lesion_id"].nunique()
                ),
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_paths: dict[str, Path] = {}
    try:
        for split_name, frame in manifests.items():
            temp_path = output_dir / f".{split_name}.csv.tmp"
            frame.to_csv(temp_path, index=False)
            temp_paths[split_name] = temp_path
        test_manifest_hash = sha256_file(temp_paths["test"])

        source_digest = sha256_file(source_metadata)
        source_filename = source_metadata.relative_to(project_root).as_posix()
        metadata = {
            "source_metadata_filename": source_filename,
            "source_metadata_sha256": source_digest,
            "image_directory": image_dir.relative_to(project_root).as_posix(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": int(seed),
            "split_method": (
                "Two-stage stratified random split of unique lesion_id groups; "
                "all images from a lesion are assigned to one split."
            ),
            "ratios_requested_by_lesion": ratios,
            "label_definition": {
                "column": DIAGNOSIS_COLUMN,
                "melanoma": {
                    "value": MELANOMA_DIAGNOSIS,
                    "label": 1,
                    "label_name": "mel",
                },
                "non_melanoma_values": sorted(NON_MELANOMA_DIAGNOSES),
                "non_melanoma_label": 0,
                "non_melanoma_label_name": "non_mel",
                "unknown_excluded_values": {
                    INDETERMINATE_DIAGNOSIS: int(
                        (
                            source[DIAGNOSIS_COLUMN]
                            == INDETERMINATE_DIAGNOSIS
                        ).sum()
                    ),
                    **unknown_values,
                },
                "unrecognized_values_are_excluded": True,
            },
            "source_counts": {
                "rows_total": int(len(source)),
                "rows_melanoma": source_class_counts["mel"],
                "rows_non_melanoma": source_class_counts["non_mel"],
                "rows_unknown_excluded": int(len(unknown_rows)),
                "unknown_rows_by_diagnosis": unknown_values,
                "lesions_melanoma": source_lesion_class_counts["mel"],
                "lesions_non_melanoma": source_lesion_class_counts["non_mel"],
                "lesions_with_unknown_diagnosis": int(
                    unknown_rows["lesion_id"].nunique()
                ),
                "duplicate_image_id_rows": duplicate_image_rows,
                "duplicate_lesion_rows": int(
                    source["lesion_id"].duplicated(keep=False).sum()
                ),
                "duplicate_lesion_groups": int(
                    (source["lesion_id"].value_counts() > 1).sum()
                ),
            },
            "splits": split_summary,
            "test_manifest_sha256": test_manifest_hash,
            "test_set_usage_policy": (
                "Held out by this preprocessing script; do not use for training, "
                "threshold selection, hyperparameter selection, or checkpoint selection."
            ),
            "validation_checks": validation_checks,
        }
        metadata_temp = output_dir / ".split_metadata.json.tmp"
        metadata_temp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for split_name, temp_path in temp_paths.items():
            os.replace(temp_path, output_dir / f"{split_name}.csv")
        os.replace(metadata_temp, output_dir / "split_metadata.json")
    finally:
        for temp_path in [*temp_paths.values(), output_dir / ".split_metadata.json.tmp"]:
            temp_path.unlink(missing_ok=True)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = prepare_dataset_splits(seed=args.seed, overwrite=args.overwrite)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
