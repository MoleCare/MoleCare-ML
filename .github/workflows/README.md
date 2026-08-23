# Workflows

`pr-checks.yml` is the only workflow in this repository. It runs on pull
requests and does not touch any cloud account:

| Job | What it does |
|---|---|
| Secret scan | gitleaks over the full history |
| Sensitive data check | blocks credential files, embedded base64 blobs, and oversized files; warns on notebook output images |
| Lint and hooks | pre-commit |
| Tests | installs `requirements.lock` and runs the suite; verifies the lock is in sync with `requirements.txt` |
| Docker build | builds `deploy/Dockerfile.web` |

It uses the `pull_request` trigger, **not** `pull_request_target`, so pull
requests from forks run with no access to repository secrets.

## Deployment

Build, registry push, and deployment happen from a separate internal
repository. They are deliberately not here:

- a public repository that deploys on merge to `main` turns every merged
  contribution into a production release
- deployment workflows encode account identifiers, role ARNs, registry names
  and environment topology that do not belong in public

`deploy/` and `Dockerfile.lambda` remain in this repository so the images are
reproducible and reviewable. Only the pipeline that ships them is elsewhere.
