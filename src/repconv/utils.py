"""Housekeeping for long runs on a shared GPU machine."""

import gc
import shutil
from pathlib import Path

import torch


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def drop_hf_cache(repo_id: str) -> None:
    """Delete downloaded weights of a Hugging Face model to keep the disk from filling up."""
    try:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
    except ImportError:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
    shutil.rmtree(cache / ("models--" + repo_id.replace("/", "--")), ignore_errors=True)


def slug(name: str) -> str:
    """File-system friendly model name."""
    return name.replace("/", "__")
