"""The GPT model — a character-level decoder-only Transformer.

This is the file you write. Build it up from the smallest piece:

    Head                 one head of causal self-attention
    MultiHeadAttention   several heads in parallel, concatenated
    FeedForward          a small per-token MLP
    Block                one Transformer layer: attention + FFN, each with a
                         residual connection and pre-LayerNorm
    GPT                  embedding -> N x Block -> LayerNorm -> lm_head

Shape convention used throughout:
    B = batch size, T = time / block_size (sequence length), C = n_embd (channels)

`tests/test_model.py` is your spec — run `pytest` as you go and fill in until green.
See `examples/build_gpt.ipynb` for the intuition behind each piece.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """All hyper-parameters in one place; train / sample / checkpoint share it."""

    vocab_size: int          # number of distinct characters (char-level vocab)
    block_size: int = 128    # context length T: how many past chars the model sees
    n_embd: int = 192        # embedding / channel dimension C
    n_head: int = 6          # number of attention heads (must divide n_embd)
    n_layer: int = 6         # how many Blocks to stack
    dropout: float = 0.1


class Head(nn.Module):
    """One head of causal self-attention.

    Maps (B, T, C) -> (B, T, head_size). Each token forms a query, scores it
    against every past token's key, softmaxes into weights, and takes the
    weighted sum of their values.
    """

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        # TODO: key / query / value linear projections (n_embd -> head_size, no bias)
        # TODO: register the lower-triangular causal mask as a buffer named "tril"
        #       (it isn't trained, but must follow the model via .to() / state_dict)
        # TODO: dropout
        raise NotImplementedError

    def forward(self, x):
        # TODO: compute k, q, v
        # TODO: attention scores = q @ k^T scaled by 1/sqrt(head_size)  -> (B, T, T)
        # TODO: mask future positions with tril (set to -inf), softmax, dropout
        # TODO: return weighted sum of values  -> (B, T, head_size)
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: head_size = n_embd // n_head
        # TODO: an nn.ModuleList of n_head Heads
        # TODO: output projection (n_embd -> n_embd) and dropout
        raise NotImplementedError

    def forward(self, x):
        # TODO: concat all heads along the channel dim, then project + dropout
        raise NotImplementedError


class FeedForward(nn.Module):
    """Position-wise feed-forward: Linear -> ReLU -> Linear, with a 4x hidden expansion."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: nn.Sequential(Linear(n_embd, 4*n_embd), ReLU, Linear(4*n_embd, n_embd), Dropout)
        raise NotImplementedError

    def forward(self, x):
        # TODO: run x through the network
        raise NotImplementedError


class Block(nn.Module):
    """One Transformer layer with pre-LayerNorm and residual connections:
    x = x + attn(ln1(x)); x = x + ffn(ln2(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: MultiHeadAttention, FeedForward, two LayerNorms
        raise NotImplementedError

    def forward(self, x):
        # TODO: residual around attention, then residual around feed-forward
        raise NotImplementedError


class GPT(nn.Module):
    """A full character-level GPT.

    forward(idx, targets) -> (logits, loss)
    generate(idx, max_new_tokens) -> autoregressive continuation
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # TODO: token embedding (vocab_size -> n_embd)
        # TODO: position embedding (block_size -> n_embd)
        # TODO: nn.Sequential of n_layer Blocks
        # TODO: final LayerNorm and lm_head (n_embd -> vocab_size)
        # TODO: self.apply(self._init_weights)
        raise NotImplementedError

    @staticmethod
    def _init_weights(module):
        # TODO (optional but recommended): normal(0, 0.02) init for Linear/Embedding weights,
        #       zero-init Linear biases
        raise NotImplementedError

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        # idx is (B, T) of int token ids
        # TODO: token embedding + position embedding  -> (B, T, C)
        # TODO: run through blocks, final LayerNorm, lm_head  -> logits (B, T, vocab_size)
        # TODO: if targets is None: return (logits, None)
        # TODO: else flatten to (B*T, vocab) and (B*T,), compute cross_entropy, return (logits, loss)
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Given a (B, T) context, autoregressively sample max_new_tokens chars."""
        # TODO: loop max_new_tokens times:
        #   - crop idx to the last block_size tokens (position embedding has no entry beyond it)
        #   - forward, take logits at the last step, divide by temperature
        #   - optional top_k filtering
        #   - softmax -> multinomial sample -> append to idx
        # TODO: return idx
        raise NotImplementedError
