# Melanoma Project Status

## Active architecture

Input image → Image Validation → Calibration → UltraLight VM-UNet →
ABCDE Feature Extraction → ViT Classification → Agentic RAG → Report
Generation.

The active segmentation checkpoint is
`models/segmentation/UltraLight_VM_UNet.pth`. It is loaded with strict
checkpoint matching. Loading or inference errors stop analysis; there is no
SAM2, OpenCV, Otsu, or morphology-based lesion-segmentation fallback. ABCDE
features are calculated from the actual VM-UNet mask.

The ViT is the primary classifier. Agentic RAG receives the ViT result and
structured ABCDE features, plans targeted queries, evaluates evidence
sufficiency, executes at most two retrieval rounds, and records provenance.
Agentic RAG does not modify the ViT prediction. If it fails, the conventional
RAG fallback metadata and evidence are retained where that fallback succeeds.

## Models and fixed manifests

- ViT backbone: `google/vit-base-patch16-224`.
- Live ViT checkpoint: `models/classification/vit_model_experiment`.
- Candidate checkpoint: `models/classification/vit_candidate_run_001`; this
  run is unfinished and isolated from serving.
- Segmentation: UltraLight VM-UNet only.
- Canonical manifests, preserved without membership changes:
  - `train.csv`: 8,110 images / 6,117 lesions.
  - `validation.csv`: 1,728 images / 1,311 lesions.
  - `test.csv`: 1,733 images / 1,311 lesions.
- The fixed manifests have no image-ID or lesion-ID overlap across splits.

The controlled candidate-training workflow uses seed 42, batch size 8, five
epochs, learning rate `1e-5`, weight decay `1e-4`, a frozen backbone, weighted
cross entropy, ordinary shuffled sampling, and validation PR-AUC for checkpoint
selection. It does not use `WeightedRandomSampler` or the test split for model
selection.

## Historical artifacts

The `sam2/` source directory, `models/segmentation/sam2/`, and
`outputs/sam2_batch/` are retained as historical experiments only. They are
not imported or called by active application runtime.

Previously generated ViT and contrastive evaluation results are historical
artifacts, not current/final performance claims. No new performance metrics
were generated during this consistency cleanup.

## Pending work

- Final ViT test evaluation.
- Larger controlled VM-UNet evaluation.
- Quantitative Agentic RAG benchmark.
- Reproducible contrastive-learning evaluation, if needed.
