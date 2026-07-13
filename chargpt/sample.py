"""Generate text from a trained charGPT checkpoint.

Example
-------
    python -m chargpt.sample --ckpt ckpt.pt --tokens 500 --prompt "ROMEO:"

The argparse plumbing and main() wiring are provided. Fill in load_model and
generate_text to match the checkpoint format you save in train.py.
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
    # TODO: torch.load the checkpoint (map_location=device)
    # TODO: GPT(ckpt["config"]) -> load_state_dict -> to(device) -> eval()
    # TODO: rebuild a CharTokenizer from ckpt["stoi"] / ckpt["itos"]
    #       (tip: CharTokenizer.__new__ lets you set stoi/itos without the original text)
    # TODO: return (model, tokenizer)
    raise NotImplementedError


def generate_text(model, tokenizer, device, prompt="", tokens=500, temperature=1.0, top_k=None):
    """Encode the prompt (or start from token 0), generate, and decode to a string."""
    # TODO: build the starting idx tensor from the prompt (or zeros((1,1)) if empty)
    # TODO: model.generate(...) then tokenizer.decode the first row
    raise NotImplementedError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample text from a charGPT checkpoint.")
    p.add_argument("--ckpt", default="ckpt.pt", help="checkpoint path")
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
