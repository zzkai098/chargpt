import torch

from chargpt.data import CharTokenizer
from chargpt.model import GPT, GPTConfig
from chargpt.sample import load_model


TEXT = "hello world, this is charGPT!\nsecond line here."


def save_ckpt(path):
    """Build a tiny model + tokenizer and save a checkpoint in train.py's format."""
    tok = CharTokenizer(TEXT)
    config = GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_embd=16,
                       n_head=4, n_layer=2, dropout=0.0)
    model = GPT(config)
    torch.save(
        {"model": model.state_dict(), "config": config,
         "stoi": tok.stoi, "itos": tok.itos},
        path,
    )
    return model, tok


def test_checkpoint_roundtrip_matches_logits(tmp_path):
    """A model reloaded from a checkpoint must produce identical logits."""
    ckpt = tmp_path / "ckpt.pt"
    original, _ = save_ckpt(ckpt)
    original.eval()

    restored, _ = load_model(str(ckpt), device="cpu")

    idx = torch.randint(0, original.config.vocab_size, (2, 8))
    with torch.no_grad():
        a, _ = original(idx)
        b, _ = restored(idx)
    assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_restores_tokenizer(tmp_path):
    """The tokenizer rebuilt from the checkpoint must round-trip text exactly."""
    ckpt = tmp_path / "ckpt.pt"
    _, tok = save_ckpt(ckpt)

    _, restored_tok = load_model(str(ckpt), device="cpu")

    assert restored_tok.decode(restored_tok.encode(TEXT)) == TEXT
    assert restored_tok.stoi == tok.stoi
