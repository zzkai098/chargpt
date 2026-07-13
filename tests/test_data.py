import torch

from chargpt.data import CharTokenizer, get_batch


TEXT = "hello world, this is charGPT!\nsecond line here."


def test_encode_decode_roundtrip():
    tok = CharTokenizer(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_size_matches_unique_chars():
    tok = CharTokenizer(TEXT)
    assert tok.vocab_size == len(set(TEXT))


def test_encode_returns_valid_ids():
    tok = CharTokenizer(TEXT)
    ids = tok.encode(TEXT)
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_get_batch_shapes_and_shift():
    tok = CharTokenizer(TEXT)
    data = torch.tensor(tok.encode(TEXT * 5), dtype=torch.long)
    block_size, batch_size = 8, 4
    x, y = get_batch(data, block_size, batch_size, device="cpu")
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # y is x shifted by one position — check it holds for the reconstructed offsets
    # (targets at position t are the next token after x's position t)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_get_batch_targets_are_next_tokens():
    """y[:, :-1] should equal x[:, 1:] because both come from a contiguous slice."""
    tok = CharTokenizer(TEXT)
    data = torch.arange(100, dtype=torch.long)  # predictable content
    torch.manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu")
    assert torch.equal(x[:, 1:], y[:, :-1])
