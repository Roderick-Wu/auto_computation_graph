from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def resolve_repo_root() -> Path:
    return REPO_ROOT


def resolve_project_root() -> Path:
    raw = os.environ.get("WRODERI_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT.parent


def resolve_scratch_root() -> Path:
    raw = os.environ.get("WRODERI_SCRATCH_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT / "scratch"


def resolve_models_root() -> Path:
    raw = os.environ.get("WRODERI_MODELS_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT / "models"


def resolve_model_path(model_name: str) -> Path:
    return resolve_models_root() / model_name


def resolve_auto_traces_root(model_name: str, experiment: str) -> Path:
    return resolve_scratch_root() / "traces" / model_name / experiment
