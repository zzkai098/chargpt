"""Character-level tokenizer and batch loader.

There is no fancy tokenization here: the vocabulary is simply the set of unique
characters in the training text. Each character maps to an integer id, and back.

`tests/test_data.py` is your spec — run `pytest` and fill in until green.
"""

from pathlib import Path

import torch


class CharTokenizer:
    """Maps characters <-> integer ids based on the characters seen in a text."""

    def __init__(self, text: str):
        # TODO: build sorted list of unique chars, then stoi and itos dicts
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        # TODO: number of distinct characters
        raise NotImplementedError

    def encode(self, s: str) -> list:
        """Turn a string into a list of int ids."""
        # TODO
        raise NotImplementedError

    def decode(self, ids) -> str:
        """Turn a list/1-D tensor of int ids back into a string."""
        # TODO
        raise NotImplementedError


def load_data(path, split: float = 0.9, device="cpu"):
    """Read a text file, build a tokenizer, and return (train, val, tokenizer).

    `train` and `val` are 1-D LongTensors of token ids on `device`.
    """
    # TODO: read the file, build a CharTokenizer, encode to a LongTensor
    # TODO: split at `split` and return (train, val, tokenizer)
    raise NotImplementedError


def get_batch(data, block_size: int, batch_size: int, device="cpu"):
    """Sample a random batch of (context, target) pairs.

    Returns x, y each of shape (batch_size, block_size). y is x shifted by one:
    the model predicts the next character at every position.
    """
    # TODO: sample batch_size random start indices in [0, len(data) - block_size)
    # TODO: x = data[i : i+block_size], y = data[i+1 : i+1+block_size], stacked
    # TODO: move to device and return (x, y)
    raise NotImplementedError
