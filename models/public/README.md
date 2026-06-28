# Public models

Curated inference models that are **safe to commit** live here.

## Workflow

1. Train and deploy from the Train YOLO tab → writes to `models/<name>/` (local testing, git-ignored).
2. When a model is ready to share, copy its folder here, e.g. `models/public/pmab_v1/`.
3. Commit only what is under `models/public/`.

Each folder needs the usual plugin layout: `model.py`, `config.yaml`, weights (`.pt`), and optional `name.txt`.
