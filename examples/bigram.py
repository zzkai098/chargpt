import torch
from torch import nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iter = 3000
eval_interval = 300
learning_rate = 1e-2
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, x, targets=None):
        logits = self.token_embedding_table(x) #(BTC)
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
            logits, loss = self(idx) #BTC
            logits = logits[:, -1, :] #BC
            probs = F.softmax(logits, dim=-1) #BC
            idx_next = torch.multinomial(probs, num_samples = 1) #B1
            idx = torch.cat([idx, idx_next], dim=1) #B T+1
        return idx
    
model = BigramLanguageModel(vocab_size)    
m = model.to(device)

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
    
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
            
            