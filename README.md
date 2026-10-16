# CEFM Backend

Cross-modal Explainable Framework for Melanoma.

## Active assessment pipeline

Input image → image validation → physical calibration → UltraLight VM-UNet
segmentation → ABCDE feature extraction from that VM-UNet mask → ViT
classification → Agentic RAG → report generation.

Segmentation uses only `models/segmentation/UltraLight_VM_UNet.pth`, loaded
strictly. If checkpoint loading or inference fails, analysis fails closed; there
is no SAM2, OpenCV, Otsu, or morphology-based lesion-mask fallback. SAM2 files
and results are historical experiments only.

The ViT supplies the primary classifier result. Agentic RAG uses that result
with structured ABCDE features for evidence retrieval and does not override the
classifier.

## Storage

Every uploaded image is saved under `data/uploads/` and its full analysis
(classification, ABCDE features incl. diameter, evolution, report) is stored
in the SQLite database `cefm.db` for future use.

## API (`/api/v1`)

| Method | Endpoint | Description |
|---|---|---|
| GET  | `/health` | Health check |
| POST | `/analyze` | Upload image + run full pipeline. Optional `case_id` / `case_label` form fields enable evolution tracking |
| POST | `/cases` | Create a case |
| GET  | `/cases` | List cases |
| GET  | `/cases/{case_id}` | Case detail with all analyses |
| GET  | `/analyses` | Recent analysis history |
| GET  | `/analyses/{id}` | Full stored result |
| GET  | `/images/{analysis_id}` | Retrieve a stored image |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
     -F "image=@lesion.jpg" -F "case_label=Left arm mole"
```

## Training

The controlled ViT workflow uses the fixed lesion-separated HAM10000 manifests
and saves an isolated candidate under `models/classification/vit_candidate_run_001`.
It does not overwrite the live application checkpoint:

```bash
python scripts/train_classifier_fixed.py
```

The live service checkpoint is `models/classification/vit_model_experiment`.
Final ViT test evaluation is pending; existing evaluation artifacts are
historical and are not current performance claims.

## Important

Do not use classifier outputs as medical diagnoses.

## Patient image quality and D measurement

Patient-uploaded images require a standardized **20 mm x 20 mm calibration marker** placed beside the lesion on the same skin surface. Images are rejected when required quality/calibration checks fail. The patient-reported approximate size is not used to calculate Diameter (D).

D is calculated from the segmented lesion's maximum pixel diameter and the marker-derived pixels-per-millimetre scale. No arbitrary pixel-to-mm fallback is used.

### Backend environment

Install `requirements.txt` in the backend virtual environment and run:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Streamlit environment

Use a separate virtual environment for Streamlit because the frontend and the pinned FastAPI backend have incompatible Starlette requirements.

```bash
python -m venv venv_frontend
venv_frontend\\Scripts\\activate
pip install -r requirements-streamlit.txt
python -m streamlit run streamlit_app.py
```

### Tests

```bash
python -m pytest -q
```
