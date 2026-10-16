import os
import json
import numpy as np
import pandas as pd

ROOT = "/mnt/d/pytest_cache"

UNCERTAINTY_FILE = os.path.join(
    ROOT,
    "outputs/sam2_batch/uncertainty_analysis/vmunet_uncertainty_20.json"
)

SAM2_FILE = os.path.join(
    ROOT,
    "outputs/sam2_batch/safe_points_results/safe_points_20_results.json"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs/sam2_batch/uncertainty_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

with open(UNCERTAINTY_FILE, "r") as f:
    uncertainty = json.load(f)

with open(SAM2_FILE, "r") as f:
    sam2_data = json.load(f)


# Handle either list or dictionary format
if isinstance(uncertainty, dict):
    if "results" in uncertainty:
        uncertainty = uncertainty["results"]
    elif "images" in uncertainty:
        uncertainty = uncertainty["images"]

if isinstance(sam2_data, dict):
    if "results" in sam2_data:
        sam2_results = sam2_data["results"]
    elif "images" in sam2_data:
        sam2_results = sam2_data["images"]
    else:
        sam2_results = []
else:
    sam2_results = sam2_data


# ---------------------------------------------------------
# BUILD LOOKUP
# ---------------------------------------------------------

uncertainty_lookup = {}

for item in uncertainty:
    name = item.get("image") or item.get("filename")
    if name:
        uncertainty_lookup[name] = item


sam2_lookup = {}

for item in sam2_results:
    name = item.get("image") or item.get("filename")
    if name:
        sam2_lookup[name] = item


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

rows = []

for name in sorted(set(uncertainty_lookup) & set(sam2_lookup)):

    u = uncertainty_lookup[name]
    s = sam2_lookup[name]

    vm_iou = float(
        s.get("vm_iou", s.get("vmunet_iou", 0))
    )

    sam_iou = float(
        s.get(
            "selected_gt_iou",
            s.get("sam2_gt_iou", s.get("gt_iou", 0))
        )
    )

    improvement = sam_iou - vm_iou

    row = {
        "image": name,

        "vm_iou": vm_iou,
        "sam2_iou": sam_iou,
        "improvement": improvement,

        "foreground_ratio": float(
            u.get("foreground_ratio", 0)
        ),

        "mean_probability": float(
            u.get("mean_probability", 0)
        ),

        "mean_confidence": float(
            u.get("mean_confidence", 0)
        ),

        "uncertain_05": float(
            u.get("uncertain_ratio_05", 0)
        ),

        "uncertain_10": float(
            u.get("uncertain_ratio_10", 0)
        ),

        "mean_entropy": float(
            u.get("mean_entropy", 0)
        ),

        "foreground_probability": float(
            u.get("mean_foreground_probability", 0)
        ),

        "foreground_confidence": float(
            u.get("foreground_confidence", 0)
        ),

        "background_probability": float(
            u.get("mean_background_probability", 0)
        ),

        "background_confidence": float(
            u.get("background_confidence", 0)
        ),
    }

    rows.append(row)


df = pd.DataFrame(rows)

print()
print("=" * 75)
print("SAM2 GATE vs VM-UNET UNCERTAINTY")
print("=" * 75)

print(f"Images matched: {len(df)}")

if len(df) == 0:
    print("\nERROR: No matching images found.")
    print("Check the JSON structures.")
    raise SystemExit


# ---------------------------------------------------------
# BASIC RESULT
# ---------------------------------------------------------

df["sam2_better"] = df["improvement"] > 0

print()
print(
    f"SAM2 improved : {df['sam2_better'].sum()}/{len(df)}"
)

print(
    f"SAM2 degraded/equal : "
    f"{(~df['sam2_better']).sum()}/{len(df)}"
)

print()
print(
    f"Mean VM IoU   : {df['vm_iou'].mean()*100:.2f}%"
)

print(
    f"Mean SAM2 IoU : {df['sam2_iou'].mean()*100:.2f}%"
)

print(
    f"Mean change   : {df['improvement'].mean()*100:+.2f} points"
)


# ---------------------------------------------------------
# CORRELATIONS
# ---------------------------------------------------------

features = [
    "foreground_ratio",
    "mean_probability",
    "mean_confidence",
    "uncertain_05",
    "uncertain_10",
    "mean_entropy",
    "foreground_probability",
    "foreground_confidence",
    "background_probability",
    "background_confidence",
    "vm_iou",
]

print()
print("=" * 75)
print("CORRELATION WITH SAM2 IMPROVEMENT")
print("=" * 75)

correlations = []

for feature in features:

    corr = df[feature].corr(df["improvement"])

    correlations.append(
        {
            "feature": feature,
            "correlation": corr
        }
    )

correlations_df = pd.DataFrame(correlations)

correlations_df = correlations_df.sort_values(
    "correlation",
    key=lambda x: abs(x),
    ascending=False
)

for _, row in correlations_df.iterrows():

    print(
        f"{row['feature']:28s} "
        f"{row['correlation']:+.4f}"
    )


# ---------------------------------------------------------
# GROUP ANALYSIS
# ---------------------------------------------------------

print()
print("=" * 75)
print("UNCERTAINTY GROUP ANALYSIS")
print("=" * 75)


def group_analysis(feature, thresholds):

    print()
    print(f"Feature: {feature}")

    for threshold in thresholds:

        low = df[df[feature] < threshold]
        high = df[df[feature] >= threshold]

        if len(low) > 0:
            low_change = low["improvement"].mean() * 100
            low_better = low["sam2_better"].mean() * 100
        else:
            low_change = np.nan
            low_better = np.nan

        if len(high) > 0:
            high_change = high["improvement"].mean() * 100
            high_better = high["sam2_better"].mean() * 100
        else:
            high_change = np.nan
            high_better = np.nan

        print(
            f"  threshold {threshold:.4f}"
        )

        print(
            f"    <  threshold: "
            f"N={len(low):2d}, "
            f"change={low_change:+.2f}, "
            f"improved={low_better:.1f}%"
        )

        print(
            f"    >= threshold: "
            f"N={len(high):2d}, "
            f"change={high_change:+.2f}, "
            f"improved={high_better:.1f}%"
        )


group_analysis(
    "mean_entropy",
    [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
)

group_analysis(
    "uncertain_10",
    [0.005, 0.01, 0.015, 0.02]
)

group_analysis(
    "foreground_ratio",
    [0.10, 0.20, 0.30, 0.40, 0.50]
)

group_analysis(
    "mean_probability",
    [0.10, 0.20, 0.30, 0.40]
)


# ---------------------------------------------------------
# SORT BY IMPROVEMENT
# ---------------------------------------------------------

print()
print("=" * 75)
print("BEST SAM2 CASES")
print("=" * 75)

best = df.sort_values(
    "improvement",
    ascending=False
).head(10)

for _, r in best.iterrows():

    print(
        f"{r['image']:22s} "
        f"VM={r['vm_iou']*100:6.2f}% "
        f"SAM2={r['sam2_iou']*100:6.2f}% "
        f"CHANGE={r['improvement']*100:+7.2f} "
        f"Entropy={r['mean_entropy']:.4f}"
    )


print()
print("=" * 75)
print("WORST SAM2 CASES")
print("=" * 75)

worst = df.sort_values(
    "improvement",
    ascending=True
).head(10)

for _, r in worst.iterrows():

    print(
        f"{r['image']:22s} "
        f"VM={r['vm_iou']*100:6.2f}% "
        f"SAM2={r['sam2_iou']*100:6.2f}% "
        f"CHANGE={r['improvement']*100:+7.2f} "
        f"Entropy={r['mean_entropy']:.4f}"
    )


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

merged_path = os.path.join(
    OUTPUT_DIR,
    "sam2_uncertainty_gate_analysis_20.csv"
)

corr_path = os.path.join(
    OUTPUT_DIR,
    "sam2_uncertainty_correlations_20.csv"
)

df.to_csv(merged_path, index=False)
correlations_df.to_csv(corr_path, index=False)


print()
print("=" * 75)
print("DONE")
print("=" * 75)

print(f"Analysis CSV : {merged_path}")
print(f"Correlation CSV : {corr_path}")
