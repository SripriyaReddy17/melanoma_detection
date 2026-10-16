# Active model setup

The application uses this fixed runtime architecture:

Input image → image validation → calibration → UltraLight VM-UNet → ABCDE
features from the VM-UNet mask → ViT classification → Agentic RAG → report.

## Required model checkpoints

- Segmentation: `models/segmentation/UltraLight_VM_UNet.pth`
- Classification: `models/classification/vit_model_experiment`

The segmentation checkpoint is loaded with strict state-dictionary matching.
Missing, incompatible, or failing segmentation checkpoints/inference stop the
analysis; no alternate segmentation model or classical lesion-mask fallback is
used. In particular, SAM2 is not part of application runtime.

The ViT serving checkpoint is separate from the unfinished candidate at
`models/classification/vit_candidate_run_001`. Candidate training and
evaluation must not replace the live serving checkpoint.

## Optional integrations

- `CLIP_MODEL_DIR` can identify an optional local CLIP model.
- `DEEPSEEK_API_KEY`, `DEEPSEEK_API_URL`, and `DEEPSEEK_MODEL` configure remote
  report synthesis. If unavailable, report generation uses its local template.

## Historical experiments

The root `sam2/` source, `models/segmentation/sam2/`, and
`outputs/sam2_batch/` are retained for research comparison only. They are not
imported or called by active application runtime. Existing evaluation JSON
files are historical artifacts, not current or final model metrics.

## Verification

Run the existing test suite from the project root:

```bash
python -m pytest -q
```
