""" Add multihead-attention, ffwd, Blocks, residual Connections, LayerNorm, Dropout """
import torch
from torch import nn
from torch.nn import functional as F

#hyperparameters
TRAIN=False

batch_size = 32
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
learning_rate = 3e-4
max_iter = 3000
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200
# ---------------------

torch.manual_seed(1337)

with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocab_size = len(chars)

#map functions
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] #take a string and map to list of ints
decode = lambda l: "".join([itos[i] for i in l]) #take a list of ints and map to string

#split 
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train = data[:n]
val   = data[n:]

#data load 
def get_batch(split:str):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 

class Head(nn.Module):
    "one head of self-attention"
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #tril is not a parameter, put in buffer
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x)     #BT C = head_size
        q = self.query(x)   #BT C = head_size
        
        #compute attention "affinities"
        wei = q @ k.transpose(-2, -1)  * C**-0.5  #BTT
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
                
        v = self.value(x)   #BT C = head_size
        out = wei @ v #BTT @  #BT C ->  #BT C = head_size
        
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # BTC= (head_size * n_heads) n_embd -> BTC=n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # c = head_size * n_heads -> n_embd
        return self.dropout(self.proj(out))
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        #BTC=n_embd -> BTC=n_embd
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #layerNorm + Residual connections
        x = x + self.ffwd(self.ln2(x)) #layerNorm + Residual connections
        return x 
        
                
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) #number of embedding dimensions
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each position has its own embd vec
        
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads=n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embd)
        )
        
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #(BTC) C = n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(TC)
        x = tok_emb + pos_emb #now the token holds token identities and position it occur
                
        x = self.blocks(x)
        
        logits = self.lm_head(x) #BTC C = vocab_size
        
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        #idx is BT
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #BT = 8, since block size= 8 in the model, if don trunk the self(idx) cannot work
            logits, loss = self(idx_cond) #BTC
            logits = logits[:, -1, :] #BC
            probs = F.softmax(logits, dim=-1) #BC
            idx_next = torch.multinomial(probs, num_samples = 1) #B1
            idx = torch.cat([idx, idx_next], dim=1) #B T+1
        return idx

# ----------train-----------

model = BigramLanguageModel()    
m = model.to(device)

if TRAIN:
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    for iter in range(max_iter):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
        xb, yb = get_batch('train')        
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    # ----------save-----------
    print(sum(p.numel() for p in m.parameters() if p.requires_grad)/1e3, 'K trainable parameters')   
    torch.save(model.state_dict(), 'model.pt')
    print('saved to model.pt')
        
else:       
    m.load_state_dict(torch.load('model.pt',map_location=device))

# ----------infer-----------
m.eval()
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))             
