import torch

from chargpt.model import GPT, GPTConfig


def tiny_config(**kw):
    base = dict(vocab_size=17, block_size=8, n_embd=16, n_head=4, n_layer=2, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_shapes():
    config = tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (3, config.block_size))
    logits, loss = model(idx)
    assert logits.shape == (3, config.block_size, config.vocab_size)
    assert loss is None


def test_loss_is_scalar_with_targets():
    config = tiny_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (3, config.block_size))
    targets = torch.randint(0, config.vocab_size, (3, config.block_size))
    _, loss = model(idx, targets)
    assert loss.ndim == 0 and loss.item() > 0


def test_causal_mask_blocks_future():
    """A change to a future token must not affect earlier positions' logits."""
    config = tiny_config(dropout=0.0)
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (1, config.block_size))
    with torch.no_grad():
        base, _ = model(idx)
        altered = idx.clone()
        altered[0, -1] = (altered[0, -1] + 1) % config.vocab_size   # change last token
        changed, _ = model(altered)
    # every position except the last must be identical
    assert torch.allclose(base[:, :-1], changed[:, :-1], atol=1e-6)
    assert not torch.allclose(base[:, -1], changed[:, -1])


def test_generate_stays_in_vocab_and_grows():
    config = tiny_config()
    model = GPT(config)
    idx = torch.zeros((2, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20)
    assert out.shape == (2, 21)
    assert out.min() >= 0 and out.max() < config.vocab_size


def test_generate_handles_context_longer_than_block():
    """generate() must crop context to block_size and not index out of range."""
    config = tiny_config(block_size=8)
    model = GPT(config)
    idx = torch.zeros((1, config.block_size + 5), dtype=torch.long)  # longer than block
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape[1] == config.block_size + 5 + 3


def test_num_params_positive():
    model = GPT(tiny_config())
    assert model.num_params() == sum(p.numel() for p in model.parameters()) > 0
