import pytest
import torch
import torch.nn.functional as F

from repconv import metrics


def unit_vectors(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def bruteforce_mutual_knn(x, y, k):
    def neighbours(z, i):
        others = [(float(z[i] @ z[j]), j) for j in range(len(z)) if j != i]
        return {j for _, j in sorted(others, reverse=True)[:k]}

    shared = sum(len(neighbours(x, i) & neighbours(y, i)) for i in range(len(x)))
    return shared / (len(x) * k)


def test_matches_bruteforce():
    x, y = unit_vectors(60, 8, 0), unit_vectors(60, 8, 1)
    assert metrics.mutual_knn(x, y, 4) == pytest.approx(bruteforce_mutual_knn(x, y, 4))


def test_same_space_up_to_rotation_aligns_perfectly():
    x = unit_vectors(256, 32, 2)
    rotation, _ = torch.linalg.qr(torch.randn(32, 32, generator=torch.Generator().manual_seed(3)))
    assert metrics.mutual_knn(x, x, 5) == 1.0
    assert metrics.mutual_knn(x, x @ rotation, 5) == pytest.approx(1.0)


def test_independent_spaces_stay_near_chance():
    n, k = 1024, 10
    score = metrics.mutual_knn(unit_vectors(n, 64, 4), unit_vectors(n, 64, 5), k)
    assert score == pytest.approx(metrics.chance_level(n, k), abs=3e-3)


def test_layer_grid_finds_the_shared_layer():
    shared = unit_vectors(200, 16, 6)
    x = torch.stack([unit_vectors(200, 16, 7), shared, unit_vectors(200, 16, 8)], dim=1)
    y = torch.stack([unit_vectors(200, 16, 9), unit_vectors(200, 16, 10), shared], dim=1)
    score, layers, grid = metrics.best_layer_pair(x, y, k=5)
    assert layers == (1, 2)
    assert score == pytest.approx(1.0)
    assert grid.shape == (3, 3)
    assert grid[0, 0] == pytest.approx(metrics.mutual_knn(x[:, 0], y[:, 0], 5))


def test_shuffled_pairs_fall_to_chance():
    x = unit_vectors(1024, 64, 11)
    assert metrics.shuffled_score(x, x, k=10) == pytest.approx(
        metrics.chance_level(1024, 10), abs=3e-3
    )


def test_remove_outliers_clamps_extreme_values():
    feats = torch.ones(4, 2, 50)
    feats[0, 0, 0] = 1e6
    assert metrics.remove_outliers(feats).max().item() == pytest.approx(1.0)


def test_sanity_check_reproduces_recorded_values():
    # the same values were obtained on Kaggle with the reference implementation
    result = metrics.sanity_check()
    assert result["identical"] == 1.0
    assert result["rotated"] == pytest.approx(1.0)
    assert result["noise_0.5"] == pytest.approx(0.3616, abs=5e-4)
    assert result["independent"] == pytest.approx(0.0091, abs=5e-4)
    assert result["shuffled"] == pytest.approx(0.0093, abs=5e-4)
