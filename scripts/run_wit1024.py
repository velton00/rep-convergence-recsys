"""Language-vision alignment on WIT-1024, after Huh et al. (2024).

Extracts layer-wise features of BLOOMZ and DINOv2 models, measures mutual k-NN
alignment for every pair of models and saves the table, a summary and a figure.
Features are cached in ``<out>/features``, so an interrupted run resumes where it stopped.

    python scripts/run_wit1024.py --out results/wit1024
"""

import argparse
import json
import platform
from pathlib import Path

import pandas as pd
import timm
import torch
import transformers
from tqdm.auto import tqdm

from repconv import metrics
from repconv.data import load_wit1024
from repconv.features import (
    language_features,
    load_language_model,
    load_vision_model,
    vision_features,
)
from repconv.plotting import plot_alignment
from repconv.utils import drop_hf_cache, free_gpu, slug

LANGUAGE_MODELS = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-1b1",
    "bigscience/bloomz-1b7",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
]
VISION_MODELS = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
]
BATCH_SIZE = {"bigscience/bloomz-3b": 4, "bigscience/bloomz-7b1": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Language-vision alignment on WIT-1024")
    parser.add_argument("--out", type=Path, default=Path("results/wit1024"))
    parser.add_argument("--llm", nargs="+", default=LANGUAGE_MODELS, help="language models")
    parser.add_argument("--lvm", nargs="+", default=VISION_MODELS, help="vision models (timm)")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--keep-weights", action="store_true", help="do not delete model weights after use"
    )
    return parser.parse_args()


def extract_with_retry(model, tokenizer, texts, batch_size):
    while True:
        try:
            return language_features(model, tokenizer, texts, batch_size, progress=tqdm)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size //= 2
            free_gpu()
            print(f"out of memory, batch size {batch_size}")


def run_language_model(name: str, texts: list[str], path: Path) -> None:
    # T4 has no bf16; small models get a second try in fp32 if fp16 overflows
    dtypes = [torch.float16] if "7b" in name else [torch.float16, torch.float32]
    for dtype in dtypes:
        model, tokenizer = load_language_model(name, dtype)
        n_params = sum(p.numel() for p in model.parameters())
        try:
            feats, bpb = extract_with_retry(model, tokenizer, texts, BATCH_SIZE.get(name, 8))
        finally:
            del model
            free_gpu()
        if torch.isfinite(feats).all():
            record = {
                "feats": feats,
                "bpb": float(bpb.mean()),
                "num_params": n_params,
                "dtype": str(dtype),
            }
            torch.save(record, path)
            print(f"{name}: {feats.shape[1]} layers, bpb {bpb.mean():.3f}")
            return
        print(f"{name}: non-finite features in {dtype}")
    raise FloatingPointError(f"{name}: non-finite features")


def run_vision_model(name: str, image, n: int, path: Path, device: str) -> None:
    model, transform = load_vision_model(name, device)
    try:
        feats = vision_features(model, transform, image, n, device=device, progress=tqdm)
        n_params = sum(p.numel() for p in model.parameters())
        torch.save({"feats": feats, "num_params": n_params}, path)
        print(f"{name}: {feats.shape[1]} blocks")
    finally:
        del model
        free_gpu()


def alignment_table(
    llms: list[str], lvms: list[str], feats_dir: Path, k: int, device: str
) -> pd.DataFrame:
    vision = {}
    for name in lvms:
        record = torch.load(feats_dir / f"{slug(name)}.pt")
        vision[name] = metrics.prepare_features(record["feats"], device), record["num_params"]

    rows = []
    for name in tqdm(llms, desc="alignment"):
        record = torch.load(feats_dir / f"{slug(name)}.pt")
        x = metrics.prepare_features(record["feats"], device)
        chance = metrics.chance_level(len(x), k)
        for lvm, (y, lvm_params) in vision.items():
            score, (i, j), _ = metrics.best_layer_pair(x, y, k)
            rows.append(
                {
                    "llm": name,
                    "lvm": lvm,
                    "llm_params_b": record["num_params"] / 1e9,
                    "lvm_params_m": lvm_params / 1e6,
                    "bpb": record["bpb"],
                    "one_minus_bpb": 1 - record["bpb"],
                    "score": score,
                    "null": metrics.shuffled_score(x[:, i], y[:, j], k),
                    "chance": chance,
                    "score_to_chance": score / chance,
                    "best_llm_layer": i,
                    "n_llm_layers": x.shape[1],
                    "best_lvm_layer": j,
                    "n_lvm_layers": y.shape[1],
                }
            )
        del x
        free_gpu()
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    feats_dir = args.out / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)

    sanity = pd.Series(metrics.sanity_check(k=args.topk), name="mutual_knn")
    sanity.to_csv(args.out / "sanity_check.csv", index_label="case")
    print(sanity.round(4).to_string(), end="\n\n")

    texts, image = load_wit1024()

    for name in args.llm:
        path = feats_dir / f"{slug(name)}.pt"
        if path.exists():
            continue
        try:
            run_language_model(name, texts, path)
        except Exception as err:  # a failed model should not stop a long background run
            print(f"{name}: {type(err).__name__}: {err}")
        if not args.keep_weights:
            drop_hf_cache(name)

    for name in args.lvm:
        path = feats_dir / f"{slug(name)}.pt"
        if path.exists():
            continue
        try:
            run_vision_model(name, image, len(texts), path, args.device)
        except Exception as err:
            print(f"{name}: {type(err).__name__}: {err}")
        if not args.keep_weights:
            drop_hf_cache(f"timm/{name}")

    llms = [m for m in args.llm if (feats_dir / f"{slug(m)}.pt").exists()]
    lvms = [m for m in args.lvm if (feats_dir / f"{slug(m)}.pt").exists()]
    table = alignment_table(llms, lvms, feats_dir, args.topk, args.device)
    table.to_csv(args.out / "alignment.csv", index=False)
    plot_alignment(table, args.out / "figures" / "alignment_vs_size")

    best = table.loc[table["score"].idxmax()]
    summary = {
        "dataset": "minhuh/prh, revision wit_1024",
        "n_pairs": len(texts),
        "metric": "mutual_knn",
        "topk": args.topk,
        "chance": metrics.chance_level(len(texts), args.topk),
        "mean_null": float(table["null"].mean()),
        "llm_models": llms,
        "lvm_models": lvms,
        "llm_dtype": "torch.float16",
        "best_pair": {"llm": best["llm"], "lvm": best["lvm"], "score": float(best["score"])},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "timm": timm.__version__,
        },
    }
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pivot = table.pivot(index="llm", columns="lvm", values="score").loc[llms, lvms]
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
