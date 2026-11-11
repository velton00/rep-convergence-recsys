import numpy as np
import pytest
import timm
import torch
from PIL import Image
from timm.data import create_transform
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BloomConfig, BloomForCausalLM, PreTrainedTokenizerFast

from repconv import metrics
from repconv.features import bits_per_byte, language_features, masked_mean, vision_features


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[9.0, 9.0], [1.0, 3.0], [3.0, 5.0]]])
    mask = torch.tensor([[0, 1, 1]])
    assert torch.allclose(masked_mean(hidden, mask), torch.tensor([[2.0, 4.0]]))


def test_bits_per_byte_of_a_uniform_model():
    logits = torch.zeros(1, 4, 16)
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones(1, 4, dtype=torch.long)
    # three predicted tokens, 4 bits each, over four bytes
    assert bits_per_byte(logits, ids, mask, ["abcd"]) == pytest.approx([3.0])


def test_bits_per_byte_skips_left_padding():
    logits = torch.zeros(1, 4, 16)
    ids = torch.tensor([[0, 0, 5, 6]])
    mask = torch.tensor([[0, 0, 1, 1]])
    assert bits_per_byte(logits, ids, mask, ["ab"]) == pytest.approx([2.0])


def tiny_language_model():
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2} | {f"w{i}": i + 3 for i in range(7)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    tokenizer.padding_side = "left"
    torch.manual_seed(0)
    config = BloomConfig(vocab_size=len(vocab), hidden_size=32, n_layer=2, n_head=4)
    return BloomForCausalLM(config).eval(), tokenizer


def test_pipeline_on_tiny_models():
    n = 24
    texts = [" ".join(f"w{(i * s) % 7}" for s in range(1, 2 + i % 4)) for i in range(n)]
    model, tokenizer = tiny_language_model()
    x, bpb = language_features(model, tokenizer, texts, batch_size=5)
    assert x.shape == (n, 3, 32)
    assert np.isfinite(bpb).all()

    # padding must not change the features of a text
    alone, _ = language_features(model, tokenizer, texts[:1], batch_size=1)
    assert torch.allclose(alone[0], x[0], atol=1e-5)

    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32).eval()
    transform = create_transform(input_size=(3, 32, 32))
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)) for _ in range(n)]
    y = vision_features(vit, transform, images.__getitem__, n, batch_size=5, device="cpu")
    assert y.shape == (n, 12, 192)

    score, _, grid = metrics.best_layer_pair(
        metrics.prepare_features(x), metrics.prepare_features(y), k=3
    )
    assert grid.shape == (3, 12)
    assert 0.0 <= score <= 1.0
