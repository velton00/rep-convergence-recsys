"""Layer-wise representations of language and vision models.

Pooling follows the reference code of Huh et al. (2024): language models give the
mean over real tokens of every hidden state, vision transformers give the CLS token
after every block.
"""

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Progress = Callable[[Iterable], Iterable]


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of token states over non-padding positions: (B, T, D), (B, T) -> (B, D)."""
    m = mask.to(hidden.device).unsqueeze(-1).float()
    return (hidden.float() * m).sum(dim=1) / m.sum(dim=1)


def bits_per_byte(
    logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor, texts: Sequence[str]
) -> list[float]:
    """Next-token cross-entropy summed over each text, in bits per UTF-8 byte."""
    values = []
    for b, text in enumerate(texts):
        lg = logits[b, :-1].float()
        target = input_ids[b, 1:].to(lg.device)
        m = mask[b, :-1].to(lg.device).float()
        nats = (F.cross_entropy(lg, target, reduction="none") * m).sum().item()
        values.append(nats / math.log(2) / max(1, len(text.encode("utf-8"))))
    return values


def _dtype_kwarg(dtype: torch.dtype) -> dict:
    import transformers

    major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
    # transformers 4.56 renamed torch_dtype to dtype
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def load_language_model(name: str, dtype: torch.dtype = torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", low_cpu_mem_usage=True, **_dtype_kwarg(dtype)
    )
    return model.eval(), tokenizer


@torch.no_grad()
def language_features(
    model,
    tokenizer,
    texts: Sequence[str],
    batch_size: int = 8,
    progress: Progress | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Pooled hidden states (N, L, D) in float32 and bits-per-byte of every text."""
    device = model.get_input_embeddings().weight.device
    feats, bpb = [], []
    starts = range(0, len(texts), batch_size)
    for start in progress(starts) if progress else starts:
        batch = list(texts[start : start + batch_size])
        enc = tokenizer(batch, padding="longest", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        # with device_map="auto" the hidden states may live on different GPUs
        layers = [masked_mean(h, mask).cpu() for h in out.hidden_states]
        feats.append(torch.stack(layers, dim=1))
        bpb.extend(bits_per_byte(out.logits, ids, mask, batch))
    return torch.cat(feats), np.array(bpb)


def load_vision_model(name: str, device: str | torch.device = "cuda"):
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(name, pretrained=True).to(device).eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


@torch.no_grad()
def vision_features(
    model,
    transform,
    image: Callable[[int], Image.Image],
    n: int,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    progress: Progress | None = None,
) -> torch.Tensor:
    """CLS token after every transformer block: (N, L, D) in float32.

    ``image(j)`` returns the j-th image.
    """
    cls = {}
    hooks = [
        block.register_forward_hook(lambda _m, _i, out, idx=idx: cls.__setitem__(idx, out[:, 0]))
        for idx, block in enumerate(model.blocks)
    ]
    feats = []
    try:
        starts = range(0, n, batch_size)
        for start in progress(starts) if progress else starts:
            stop = min(start + batch_size, n)
            batch = torch.stack([transform(image(j).convert("RGB")) for j in range(start, stop)])
            model.forward_features(batch.to(device))
            feats.append(torch.stack([cls[i].float().cpu() for i in range(len(hooks))], dim=1))
    finally:
        for hook in hooks:
            hook.remove()
    return torch.cat(feats)
