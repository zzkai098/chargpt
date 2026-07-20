"""The GPT model — a character-level decoder-only Transformer.

    Head                 one head of causal self-attention
    MultiHeadAttention   several heads in parallel, concatenated
    FeedForward          a small per-token MLP
    Block                one Transformer layer: attention + FFN, each with a
                         residual connection and pre-LayerNorm
    GPT                  embedding -> N x Block -> LayerNorm -> lm_head

Shape convention used throughout:
    B = batch size, T = time / block_size (sequence length), C = n_embd (channels)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """All hyper-parameters; train / sample / checkpoint share it."""
    vocab_size:int           # number of distinct characters (char-level vocab)
    block_size:int = 256     # context length T: how many past chars the model sees
    n_embd:int = 128         # embedding / channel dimension C
    n_head:int = 4           # number of attention heads (must divide n_embd)
    n_layer:int = 6          # number of Blocks (MultiHeadAttention + FeedForward)
    dropout:float = 0.1
    
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


class Head(nn.Module):
    """One head of causal self-attention.

    Maps (B, T, C = n_embd) -> (B, T, head_size). Each token forms a query, scores it
    against every past token's key, softmaxes into weights, and takes the
    weighted sum of their values.
    """
    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.head_size = head_size
        # key / query / value linear projections (n_embd -> head_size, no bias)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))) # T,T
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) #(B, T, n_embd) -> (B, T, head_size)
        k = self.key(x)   #(B, T, n_embd) -> (B, T, head_size)
        v = self.value(x) #(B, T, n_embd) -> (B, T, head_size)
        
        # attention scores = q @ k^T scaled by 1/sqrt(head_size) -> BTT
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 
        # mask future positions with tril (set to -inf), softmax, dropout
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) # (B T T)
        wei = self.dropout(wei)

        out = wei @ v # (B T T) @ (B, T, head_size) ->   (B, T, head_size) 
        return out 
    

class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected back."""    
    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.n_head = config.n_head        
        self.head_size = head_size
    
        self.heads = nn.ModuleList([Head(config, self.head_size) for _ in range(self.n_head)])
        self.proj = nn.Linear(self.n_head * self.head_size, config.n_embd) # output projection (n_embd -> n_embd) and dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # concat all heads along the channel dim, then project + dropout
        # (B, T, head_size) -> (B, T, head_size * n_head) == (B, T, n_embd)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))
        
        
class FeedForward(nn.Module):
    """Position-wise feed-forward: Linear -> ReLU -> Linear, with a 4x hidden expansion."""    
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_embd = config.n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)
        

class Block(nn.Module):
    """One Transformer layer with pre-LayerNorm and residual connections:
    x = x + attn(ln1(x)); x = x + ffn(ln2(x))."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        # MultiHeadAttention, FeedForward, two LayerNorms
        self.sa = MultiHeadAttention(config) 
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        # residual around attention, then residual around feed-forward
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x 


class GPT(nn.Module):    
    """A full character-level GPT.

    forward(idx, targets) -> (logits, loss)
    generate(idx, max_new_tokens) -> autoregressive continuation
    """
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd) # (vocab_size -> n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd) # (block_size -> n_embd)        
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)],
            nn.LayerNorm(config.n_embd)
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size) # (n_embd -> vocab_size)
        self.apply(self._init_weights) 
        
    def forward(self, idx, targets=None):
        # idx is (B, T) of int token ids
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) # (B, T) -> (B, T, n_embd)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_embd + pos_embd # (B, T, n_embd) Now the vec contain x info and pos info
        
        x = self.blocks(x) # n_layer times (MultiHeadAttention + FeedForward) (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, n_embd) -> # (B, T, vocab_size) 
        
        if targets is None:
            loss = None
        else:
            # flatten to (B*T, vocab) and (B*T,), compute cross_entropy, return (logits, loss)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Given a (B, T) context, autoregressively sample max_new_tokens chars."""
        # idx is (B, T) of int token ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # logits  (B, T, vocab_size) 
            logits = logits[:, -1, :] / temperature #  (B, vocab_size) 
            
            # top_k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) 
            idx = torch.cat([idx, idx_next], dim=-1) # (B, T + 1) 
        
        return idx 
    
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init_weights(module):
        # normal(0, 0.02) init for Linear/Embedding weights,
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
