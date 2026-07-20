"""Generate text from a trained charGPT checkpoint.

Example
-------
    python -m chargpt.sample --ckpt checkpoints/ckpt.pt --tokens 500 --prompt "ROMEO:"

`load_model` rebuilds the GPT and tokenizer from the checkpoint written by
`train.py`; `generate_text` encodes the prompt, samples, and decodes back to text.
"""

import argparse

import torch

from .data import CharTokenizer
from .model import GPT
from .train import get_device


def load_model(ckpt_path, device):
    """Rebuild the model from a checkpoint and load its weights for inference.

    Return (model, tokenizer). Remember to put the model in eval() mode so
    dropout is off at inference time.
    """
    # weights_only=False: the checkpoint stores a GPTConfig object, which the
    # PyTorch 2.6+ safe loader rejects by default. Safe here — it's our own file.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = ckpt["stoi"]
    tokenizer.itos = ckpt["itos"]
    
    return model, tokenizer


def generate_text(model, tokenizer, device, prompt="", tokens=500, temperature=1.0, top_k=None):
    """Encode the prompt (or start from token 0), generate, and decode to a string."""
    if len(prompt) == 0:
        idx = torch.zeros((1,1), dtype=torch.long, device=device) #(1,1)
    else:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device) #(1, len)
        
    out = model.generate(idx, max_new_tokens=tokens, temperature=temperature, top_k=top_k)    
    return tokenizer.decode(out[0].tolist())
    

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample text from a charGPT checkpoint.")
    p.add_argument("--ckpt", default="checkpoints/ckpt.pt", help="checkpoint path")
    p.add_argument("--prompt", default="", help="optional starting text")
    p.add_argument("--tokens", type=int, default=500, help="how many chars to generate")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    model, tokenizer = load_model(args.ckpt, device)
    print(generate_text(
        model, tokenizer, device,
        prompt=args.prompt, tokens=args.tokens,
        temperature=args.temperature, top_k=args.top_k,
    ))


if __name__ == "__main__":
    main()
