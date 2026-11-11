"""Datasets used in the experiments."""

from collections.abc import Callable

from PIL import Image


def load_wit1024() -> tuple[list[str], Callable[[int], Image.Image]]:
    """WIT-1024 from Huh et al. (2024): the first caption of every image and an image getter."""
    from datasets import load_dataset

    ds = load_dataset("minhuh/prh", revision="wit_1024", split="train")
    texts = [str(t[0] if isinstance(t, list | tuple) else t) for t in ds["text"]]
    return texts, lambda j: ds[j]["image"]
