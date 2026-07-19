"""Train charGPT on a text file and save a checkpoint.

Example
-------
    python -m chargpt.train --data data/input.txt --steps 5000 --out ckpt.pt

The checkpoint should store the model's state_dict, the GPTConfig, and the
tokenizer maps — everything sample.py needs to rebuild and run the model.

"""

import argparse
import os

import torch

from .data import get_batch, load_data
from .model import GPT, GPTConfig


def get_device() -> torch.device:
    """mps / cuda / cpu, with an MPS CPU-fallback for unsupported ops."""
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    """Average loss over a few random batches of train and val (less noisy).

    Return a dict like {"train": float, "val": float}. Remember to switch the
    model to eval() while measuring and back to train() afterwards.
    """
    out = {}
    model.eval()
    for name, data in zip(["train", "val"], [train_data,val_data]):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(data, block_size=block_size, batch_size=batch_size, device=device)            
            logits, loss = model(xb, yb)            
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out 
            
            
def train(args):
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    train_data, val_data, tokenizer = load_data(args.data, device="cpu")
    print(f"vocab size: {tokenizer.vocab_size}  |  train tokens: {len(train_data):,}")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    print(f"model parameters: {model.num_params() / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    
    for i in range(args.steps + 1):
        if i % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, 
                                   config.block_size, args.batch_size, device)
            print(f"step {i}: train {losses['train']:.4f}  val {losses['val']:.4f}")
        
        xb, yb = get_batch(train_data, config.block_size, args.batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        
    }, args.out)
    print(f"saved checkpoint to {args.out}")
    
    
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train charGPT on a text file.")
    p.add_argument("--data", default="data/input.txt", help="path to training text")
    p.add_argument("--out", default="ckpt.pt", help="checkpoint output path")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-embd", type=int, default=192)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    return p


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
