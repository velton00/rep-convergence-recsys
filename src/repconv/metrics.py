"""Mutual k-NN alignment between two sets of paired representations.

The measure and feature preprocessing follow Huh et al. (2024),
https://github.com/minyoungg/platonic-rep.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def chance_level(n: int, k: int) -> float:
    """Expected mutual k-NN score of two independent spaces."""
    return k / (n - 1)


def knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k nearest rows of ``x`` by inner product, the row itself excluded."""
    sim = x @ x.T
    sim.fill_diagonal_(-math.inf)
    return sim.topk(k, dim=1).indices


def knn_masks(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Flattened k-NN adjacency matrices of every layer of an (N, L, D) tensor: (L, N * N)."""
    n, n_layers = feats.shape[:2]
    rows = torch.arange(n, device=feats.device).unsqueeze(1)
    masks = torch.zeros(n_layers, n, n, device=feats.device)
    for layer in range(n_layers):
        masks[layer][rows, knn_indices(feats[:, layer], k)] = 1.0
    return masks.flatten(start_dim=1)


def mutual_knn(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> float:
    """Average fraction of shared k nearest neighbours for paired rows of ``x`` and ``y``."""
    shared = knn_masks(x.unsqueeze(1), k) @ knn_masks(y.unsqueeze(1), k).T
    return shared.item() / (x.shape[0] * k)


def layer_grid(x: torch.Tensor, y: torch.Tensor, k: int = 10) -> np.ndarray:
    """Mutual k-NN for every pair of layers of ``x`` (N, Lx, D) and ``y`` (N, Ly, D')."""
    shared = knn_masks(x, k) @ knn_masks(y, k).T
    return (shared / (x.shape[0] * k)).cpu().numpy()


def best_layer_pair(
    x: torch.Tensor, y: torch.Tensor, k: int = 10
) -> tuple[float, tuple[int, int], np.ndarray]:
    """Best score over all layer pairs, the pair itself and the full grid."""
    grid = layer_grid(x, y, k)
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    return float(grid[i, j]), (int(i), int(j)), grid


def shuffled_score(
    x: torch.Tensor, y: torch.Tensor, k: int = 10, n_perm: int = 5, seed: int = 0
) -> float:
    """Mutual k-NN after randomly re-pairing the rows of ``y``, averaged over permutations."""
    g = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(n_perm):
        perm = torch.randperm(y.shape[0], generator=g).to(y.device)
        scores.append(mutual_knn(x, y[perm], k))
    return float(np.mean(scores))


def remove_outliers(feats: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """Clamp values at the q-quantile of |feats| per sample, averaged over samples.

    ``kthvalue`` replaces ``torch.quantile``, which refuses inputs above 16M elements.
    """
    flat = feats.abs().flatten(start_dim=1)
    m = flat.shape[1]
    rank = min(m, round(q * (m - 1)) + 1)
    threshold = flat.kthvalue(rank, dim=1).values.mean()
    return feats.clamp(-threshold, threshold)


def prepare_features(feats: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
    """Clamp outliers and L2-normalise every layer of an (N, L, D) tensor."""
    feats = remove_outliers(feats.float().to(device))
    return F.normalize(feats, dim=-1)


def sanity_check(n: int = 1024, d: int = 256, k: int = 10, seed: int = 0) -> dict[str, float]:
    """Mutual k-NN on synthetic cases with a known answer."""
    g = torch.Generator().manual_seed(seed)
    x = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    rotation, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    noise = torch.randn(n, d, generator=g) / math.sqrt(d)
    cases = {
        "identical": x,
        "rotated": x @ rotation,
        "noise_0.5": x + 0.5 * noise,
        "independent": torch.randn(n, d, generator=g),
        "shuffled": x[torch.randperm(n, generator=g)],
    }
    return {name: mutual_knn(x, F.normalize(y, dim=-1), k) for name, y in cases.items()}
