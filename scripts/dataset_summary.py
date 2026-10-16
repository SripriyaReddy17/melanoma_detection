"""
Dataset inspection and summary script for CEFM.
Audits the actual HAM10000 and ISIC2018 datasets on disk.
Reports:
- total images
- total masks
- total lesions
- class distribution
- melanoma count
- benign/non-melanoma count
- missing images
- missing masks
- duplicate images
- train/validation/test counts
- lesion-level leakage checks
"""

import os
import sys
from pathlib import Path
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent

def inspect_datasets():
    print("=" * 60)
    print("CEFM DATASET INSPECTION & SUMMARY AUDIT")
    print("=" * 60)
    
    # ---------------------------------------------------------
    # 1. HAM10000 Classification Dataset Audit
    # ---------------------------------------------------------
    print("\n--- [1] HAM10000 CLASSIFICATION DATASET ---")
    ham_root = BASE_DIR / "datasets" / "Ham10000"
    meta_path = ham_root / "HAM10000_metadata.csv"
    
    if not meta_path.exists():
        print(f"[ERROR] HAM10000 metadata file not found at {meta_path}")
        ham_summary = {"status": "missing"}
    else:
        df = pd.read_csv(meta_path)
        total_meta_images = len(df)
        total_unique_lesions = df["lesion_id"].nunique()
        
        # Check image files on disk
        img_dirs = [
            ham_root / "HAM10000_images_part_1",
            ham_root / "HAM10000_images_part_2",
        ]
        disk_images = {}
        duplicates = 0
        for d in img_dirs:
            if d.exists():
                for p in d.glob("*.jpg"):
                    stem = p.stem
                    if stem in disk_images:
                        duplicates += 1
                    disk_images[stem] = p
        
        total_disk_images = len(disk_images)
        missing_images = sum(1 for img_id in df["image_id"] if img_id not in disk_images)
        
        # Class distribution
        class_dist = df["dx"].value_counts().to_dict()
        melanoma_count = int(class_dist.get("mel", 0))
        benign_count = total_meta_images - melanoma_count
        
        print(f"Total metadata records: {total_meta_images}")
        print(f"Total image files discovered on disk: {total_disk_images}")
        print(f"Total unique lesions: {total_unique_lesions}")
        print(f"Missing images referenced in metadata: {missing_images}")
        print(f"Duplicate image filenames across folders: {duplicates}")
        print(f"Multi-class distribution: {class_dist}")
        print(f"Binary classification distribution:")
        print(f"  - Class 1 (Melanoma / 'mel'): {melanoma_count} ({melanoma_count/total_meta_images:.2%})")
        print(f"  - Class 0 (Benign / non-melanoma): {benign_count} ({benign_count/total_meta_images:.2%})")
        
        # Check splits
        splits_dir = ham_root / "processed"
        train_csv = splits_dir / "train.csv"
        val_csv = splits_dir / "validation.csv"
        test_csv = splits_dir / "test.csv"
        
        split_counts = {}
        leakage_detected = False
        if train_csv.exists() and val_csv.exists() and test_csv.exists():
            tr = pd.read_csv(train_csv)
            val = pd.read_csv(val_csv)
            ts = pd.read_csv(test_csv)
            
            tr_les = set(tr["lesion_id"])
            val_les = set(val["lesion_id"])
            ts_les = set(ts["lesion_id"])
            
            tr_val_overlap = len(tr_les & val_les)
            tr_ts_overlap = len(tr_les & ts_les)
            val_ts_overlap = len(val_les & ts_les)
            
            split_counts = {
                "train": len(tr),
                "train_mel": int((tr["label"] == 1).sum()),
                "train_benign": int((tr["label"] == 0).sum()),
                "validation": len(val),
                "val_mel": int((val["label"] == 1).sum()),
                "val_benign": int((val["label"] == 0).sum()),
                "test": len(ts),
                "test_mel": int((ts["label"] == 1).sum()),
                "test_benign": int((ts["label"] == 0).sum()),
                "train_val_lesion_overlap": tr_val_overlap,
                "train_test_lesion_overlap": tr_ts_overlap,
                "val_test_lesion_overlap": val_ts_overlap,
            }
            if tr_val_overlap > 0 or tr_ts_overlap > 0 or val_ts_overlap > 0:
                leakage_detected = True
                
            print(f"Split breakdown (Lesion-level disjoint splits):")
            print(f"  - Train: {len(tr)} images ({split_counts['train_mel']} mel, {split_counts['train_benign']} benign)")
            print(f"  - Validation: {len(val)} images ({split_counts['val_mel']} mel, {split_counts['val_benign']} benign)")
            print(f"  - Test: {len(ts)} images ({split_counts['test_mel']} mel, {split_counts['test_benign']} benign)")
            print(f"  - Lesion Leakage Check: {'LEAKAGE DETECTED' if leakage_detected else 'PASSED (0 overlap)'}")
        else:
            print("[WARNING] Processed splits directory not found.")
            
    # ---------------------------------------------------------
    # 2. ISIC2018 Segmentation Dataset Audit
    # ---------------------------------------------------------
    print("\n--- [2] ISIC2018 LESION SEGMENTATION DATASET ---")
    isic_root = BASE_DIR / "datasets" / "isic2018" / "data"
    isic_images_dir = isic_root / "images"
    isic_masks_dir = isic_root / "annotations"
    
    total_isic_images = 0
    total_isic_masks = 0
    missing_masks = 0
    isic_splits = {}
    
    for split in ["train", "val", "test"]:
        s_img_dir = isic_images_dir / split
        s_mask_dir = isic_masks_dir / split
        
        imgs = {p.stem: p for p in s_img_dir.glob("*.*")} if s_img_dir.exists() else {}
        # ISIC masks often have suffix '_segmentation' or same stem
        masks = {p.stem.replace("_segmentation", ""): p for p in s_mask_dir.glob("*.*")} if s_mask_dir.exists() else {}
        
        split_missing = sum(1 for stem in imgs if stem not in masks)
        missing_masks += split_missing
        total_isic_images += len(imgs)
        total_isic_masks += len(masks)
        
        isic_splits[split] = {
            "images": len(imgs),
            "masks": len(masks),
            "missing_masks": split_missing,
        }
        print(f"ISIC {split.upper()} Split: {len(imgs)} images, {len(masks)} masks (missing: {split_missing})")
        
    print(f"Total ISIC images across splits: {total_isic_images}")
    print(f"Total ISIC masks across splits: {total_isic_masks}")
    print(f"Total missing mask pairings: {missing_masks}")
    
    print("\n" + "=" * 60)
    print("DATASET AUDIT COMPLETED")
    print("=" * 60)
    
    summary = {
        "ham10000": {
            "total_images": total_meta_images,
            "total_disk_images": total_disk_images,
            "total_unique_lesions": total_unique_lesions,
            "melanoma_count": melanoma_count,
            "benign_count": benign_count,
            "missing_images": missing_images,
            "duplicates": duplicates,
            "class_distribution": class_dist,
            "splits": split_counts,
            "leakage_free": not leakage_detected,
        },
        "isic2018": {
            "total_images": total_isic_images,
            "total_masks": total_isic_masks,
            "missing_masks": missing_masks,
            "splits": isic_splits,
        }
    }
    return summary

if __name__ == "__main__":
    inspect_datasets()
