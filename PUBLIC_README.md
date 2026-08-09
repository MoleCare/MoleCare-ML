# MoleCare-ML

Flask / TensorFlow service for **research and educational** mole-image analysis used by the [MoleCare](https://www.molecare.co.uk/) skin-health apps.

> **Not a medical device.** Predictions and ABCDE helpers are **not** diagnoses. Always seek care from a qualified clinician for concerning skin changes.

---

## Features

| Endpoint area | What it does |
|---------------|--------------|
| `/predict` | Melanoma vs not-melanoma score (baseline CNN) |
| `/analyze`, `/analyze/abcde` | Structured analysis + ABCDE-oriented CV signals |
| `/detect` | Lesion detection helpers |
| `/evolution` | Temporal comparison between images |
| `/predict-advanced`, `/compare-models` | Multi-model / premium paths (optional) |
| `/health` | Liveness |

Optional: Google [Derm Foundation](https://huggingface.co/google/derm-foundation) embeddings (gated model; requires Hugging Face token + acceptance of Google Health AI terms).

---

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place or download a SavedModel under ./cnn-models/xception/1/  (see Releases)
export MODEL_PATH=./cnn-models/xception/1
gunicorn --bind 0.0.0.0:5000 --timeout 300 wsgi
```

Docker:

```bash
docker build -t molecare-ml .
docker run --rm -p 5000:5000 -e PORT=5000 molecare-ml
curl http://localhost:5000/health
```

Compose (nginx + app) may be available under `deploy/` — use localhost only; do not bake cloud credentials into images.

---

## Configuration (env only)

| Variable | Purpose |
|----------|---------|
| `MODEL_PATH` | Path to TensorFlow SavedModel |
| `PORT` | HTTP port (default 5000) |
| `HUGGINGFACE_TOKEN` | Optional Derm Foundation access |
| `WANDB_API_KEY` | Optional training logging |
| `AWS_*` | Optional deploy tooling — use IAM roles / local profile, **never** commit keys |

---

## Training (optional)

- Experiment notebooks live under `training-notebooks/`  
- Metaflow flow: `flows/training_flow.py` (set your own S3 bucket via env/flags)  
- Public derm datasets (e.g. Kaggle) have **their own licenses** — document provenance before redistributing weights or images  

Do **not** commit `kaggle.json`, AWS keys, or patient photos.

---

## API sketch

```bash
# Health
curl -s http://localhost:5000/health

# Predict (multipart image) — shape depends on your deployed handlers
curl -s -X POST http://localhost:5000/predict \
  -F "file=@sample.jpg"
```

See `ml_model_serving/` for route definitions and response schemas. Responses should include a non-diagnostic disclaimer.

---

## Models & licenses

| Component | Notes |
|-----------|--------|
| Xception / ImageNet-initialized Keras backbones | Follow TensorFlow / Keras license terms |
| Google Derm Foundation | [Terms](https://developers.google.com/health-ai-developer-foundations/terms); HF gated |
| Training data | Cite dataset sources (ISIC / HAM10000 / your Kaggle dataset) and redistribution rules |

Publish large weights via **GitHub Releases** or object storage — avoid committing multi‑MB binaries if possible.

---

## Intended use & limitations

- **Intended:** research, education, product prototyping behind MoleCare’s own clinical disclaimers  
- **Not intended:** autonomous diagnosis, triage without a clinician, or regulatory claims  
- Performance varies by skin type, lighting, image quality, and dataset bias — evaluate before any production use  

---

## Security

- Never commit PEM files, AWS keys, or `.env`  
- Rotate any credential that ever appeared in git history  
- Keep nested product docs / runbooks out of this repository  

See the MoleCare DevBox doc: `docs/OPEN_SOURCE_SANITIZE_CHECKLIST.md`.

---

## Related

- [MoleCare](https://www.molecare.co.uk/)  
- MoleCare MCP server (assistant / ops tools)  
- Mobile apps on [App Store](https://apps.apple.com/us/app/molecare/id1448635328) and [Google Play](https://play.google.com/store/apps/details?id=com.mymolecare)

---

## License

Add an OSS license (Apache-2.0 or MIT) before publishing. Third-party model/dataset licenses still apply.

---

### Maintainer note (remove before publish)

This is a **draft** public README. **Do not flip the repo public** until:

1. AWS / EC2 / Kaggle credentials are **rotated**  
2. `deploy/*.pem` removed from git **and** history  
3. `molecare-docs/` excluded  
4. Checklist in `docs/OPEN_SOURCE_SANITIZE_CHECKLIST.md` §3 signed off  
5. `mv PUBLIC_README.md README.md`  
