# Contributing

Thanks for your interest. This repository holds the machine-learning service behind the
[MoleCare](https://www.molecare.co.uk/) skin-health apps — a TensorFlow melanoma classifier
served over Flask, plus the training notebooks behind it.

## Before you start — clinical safety

This is **not a medical device**, and no contribution may present it as one. Pull requests that
remove or weaken the non-diagnostic disclaimers in the API responses or documentation will not
be merged. If a change affects how a prediction is presented to an end user, say so explicitly
in the pull request.

## Before your first commit — set up the safety hooks

This repository handles medical-adjacent data and has previously had credentials committed to it.
The hooks below run **before** a commit is created, which is the only point where a mistake is
cheap to fix. Once something is in git history, removing it means rewriting the repository.

```bash
pip install pre-commit
pre-commit install
```

That's it — the hooks now run on every `git commit`. To check the whole tree at once:

```bash
pre-commit run --all-files
```

### What the hooks stop

| Hook | Catches |
|---|---|
| `detect-private-key` | RSA / EC / OpenSSH private keys |
| `gitleaks` | AWS keys, service-account JSON, tokens, high-entropy strings |
| `nbstripout` | Jupyter outputs — **the most common way data leaks out of an ML repo**, because outputs embed images and dataframes as base64 |
| `check-added-large-files` | Anything over 5 MB — model weights belong in Releases |
| `ruff` | Lint and formatting |

`training-notebooks/` is excluded from `nbstripout` because its plots are reviewed published
results. Any **new** notebook gets stripped.

### The same checks run on your pull request

CI re-runs them, so a hook you skipped locally is caught before merge. Pull request checks run
with **no access to repository secrets** — a fork PR cannot reach our AWS or W&B credentials,
by design. Deployment jobs are gated to pushes on `main`.

If CI flags a file, do not just delete it in a new commit — it stays in history. Say so in the
pull request and a maintainer will help rewrite the branch.

## Data rules

- **Never** commit patient, user, or clinical images. Only openly licensed dermoscopic data
  (for example the ISIC Archive), with provenance documented.
- **Never** commit credentials: no `.pem`, `.key`, service-account JSON, `kaggle.json`, `.env`.
  CI blocks these paths outright.
- If you add training data, record its licence in [MODEL_CARD.md](MODEL_CARD.md).

## Infrastructure and AWS

**Never run `terraform apply` against MoleCare's AWS account.** Contributors have no access to
it, and nothing in this repository should give them any. If you want to try an infrastructure
change, run it **in your own fork, against your own AWS account** — fork the repo, point the
Terraform variables at your account, and apply there. Never open a PR that expects a maintainer
to apply infrastructure on your behalf without discussing it in an issue first.

What this means in practice:

- **No AWS credentials are stored in this repository.** There are no `AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` secrets and none should ever be added. CI authenticates with
  short-lived credentials via GitHub OIDC, so there is no long-lived key to leak.
- **CI is blocked from running `terraform apply` at the IAM level.** The role CI assumes
  (`MoleCareGitHubActionsRole`) carries an explicit `Deny` on the Terraform state bucket and
  lock table, and on `iam:*` / `ec2:*` / `rds:*` and resource-lifecycle actions. An explicit
  `Deny` overrides every `Allow` in IAM, so this holds even if the role's permissions are
  widened later. Terraform cannot apply without reading and locking remote state, so it fails
  at init. CI can push images to ECR and update `molecare-ml-*` Lambda code and aliases —
  deploying code onto infrastructure that already exists. Provisioning is always a
  `terraform apply` run by a human with their own credentials.
- **Pull requests from forks get nothing.** `pr-checks.yml` uses the `pull_request` trigger, so
  fork PRs run with a read-only token, no secrets, and no OIDC token. That is deliberate — see
  the comment at the top of that workflow. Do not change it to `pull_request_target`.
- Infrastructure lives in a separate private repository (`molecare-terraform`). Infra changes
  are not reviewed here; open an issue describing what you need.

## Reporting model performance

State the evaluation set and the metrics. For this model, **accuracy alone is not sufficient** —
sensitivity matters more, because a missed melanoma is the harmful error. See
[MODEL_CARD.md](MODEL_CARD.md) for what is currently measured and what is not.

## Ways to help


- **Model quality** — improved architectures, augmentation, or calibration, with metrics to back it
- **Bias evaluation** — performance across Fitzpatrick skin types is the most valuable open problem here
- **Serving** — latency, memory, batching, model loading
- **Documentation** — the API schemas in `ml_model_serving/` are under-documented
- **Tests** — `tests/` is thin relative to the codebase

## Getting set up

Requires **Python 3.12+**.

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
