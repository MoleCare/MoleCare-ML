# Contributing

Thanks for your interest. This repository holds the machine-learning service behind the
[MoleCare](https://www.molecare.co.uk/) skin-health apps — a TensorFlow melanoma classifier
served over Flask, plus the training notebooks behind it.

## Before you start — clinical safety

This is **not a medical device**, and no contribution may present it as one. Pull requests that
remove or weaken the non-diagnostic disclaimers in the API responses or documentation will not
be merged. If a change affects how a prediction is presented to an end user, say so explicitly
in the pull request.

## Ways to help

- **Model quality** — improved architectures, augmentation, or calibration, with metrics to back it
- **Bias evaluation** — performance across Fitzpatrick skin types is the most valuable open problem here
- **Serving** — latency, memory, batching, model loading
- **Documentation** — the API schemas in `ml_model_serving/` are under-documented
- **Tests** — `tests/` is thin relative to the codebase

## Getting set up

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MODEL_PATH=./cnn-models/xception/1
gunicorn --bind 0.0.0.0:5000 --timeout 300 wsgi
```

## Rules that matter here

- **Never commit credentials** — no `.pem`, no service-account JSON, no `kaggle.json`, no `.env`.
  The `.gitignore` covers these; do not override it.
- **Never commit patient or user images.** Only openly licensed dermoscopic data (for example the
  ISIC Archive) belongs in this repository, and its provenance must be documented.
- **Report model performance honestly.** Include the evaluation set and metrics; an unqualified
  accuracy number is not useful for a clinical-adjacent model.

## Making a change

1. Fork, branch, keep the change focused.
2. Note any dataset you used and its licence.
3. Open a pull request and fill in the template.

By contributing you agree that your work is licensed under the [Apache License 2.0](LICENSE).
