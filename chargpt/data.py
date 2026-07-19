"""Character-level tokenizer and batch loader.

There is no fancy tokenization here: the vocabulary is simply the set of unique
characters in the training text. Each character maps to an integer id, and back.

`tests/test_data.py` is your spec — run `pytest` and fill in until green.
"""
import torch


class CharTokenizer:
    """Maps characters <-> integer ids based on the characters seen in a text."""
    def __init__(self, text:str):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch:i for i, ch in enumerate(self.chars)}
        self.itos = {i:ch for i, ch in enumerate(self.chars)}
                
    @property
    def vocab_size(self) -> int:
        return len(self.chars)
    
    def encode(self, s:str) -> list[int]:
        """Turn a string into a list of int id."""
        return [self.stoi[ch] for ch in s]
        
    def decode(self, ids:list) -> str:
        """Turn a list/1-D tensor of int ids back into a string."""
        return "".join([self.itos[int(i)] for i in ids])
      
        
def load_data(path, split: float = 0.9, device="cpu"):
    """Read a text file, build a tokenizer, and return (train, val, tokenizer).
    `train` and `val` are 1-D LongTensors of token ids on `device`.
    """
    assert 0 < split < 1, "split must be between 0 and 1"
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long, device=device)
    n = int(split * len(data))
    train = data[:n]
    val = data[n:]
    return train, val, tok


def get_batch(data, block_size: int, batch_size: int, device="cpu"):
    """Sample a random batch of (context, target) pairs.

    Returns x, y each of shape (batch_size, block_size). y is x shifted by one:
    the model predicts the next character at every position.
    """
    n = len(data)
    idx = torch.randint(n - block_size, (batch_size,)) 
    xb = torch.stack([data[i  : i+block_size  ] for i in idx]) #BT
    yb = torch.stack([data[i+1: i+block_size+1] for i in idx]) #BT
    return xb.to(device), yb.to(device)
