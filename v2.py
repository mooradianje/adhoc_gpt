import os, sys, datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

# Important reading:
# Attention is all you need - OG paper on transformers
# Residual blocks - building blocks of resnet -> residual block info
# pytorch docs - understanding implementation of pytorch neural nets
# "linear projection"
# nn.Linear -> what is this (dense layer?)
# nn.Sequential -> understand when to use this
# Layer Normalization peper

# hyperparameter setting
batch_size = 32 # how many independent sequences in parallel?
block_size = 8 # maximum context length for prediction
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3 # low bc self-attention can't tolerate high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu' # just in case we have cuda
device = 'cpu'
eval_iters = 200
n_embd = 32
dbg_iter = 0

print('device: '+ device)
# head_size = 16

torch.manual_seed=(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('./data/input.txt', 'r', encoding='utf8') as f:
  text = f.read()

# basic character encoding
chars = sorted(list(set(text))) # orders all characters that occur in the dataset

# vocab_size is a global var, removed from class
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate (chars)}
itos = {i:ch for i, ch in enumerate (chars)}
encode = lambda s: [stoi[c] for c in s] # input string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # input list of integers, output string


data = torch.tensor(encode(text), dtype = torch.long)


# separate dataset into training and validation set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
  # generate a small batch of daa of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  #print(ix)
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

# everything that happens inside function we will NOT call backwards on (@torch.no_grad())
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class LayerNorm:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps

    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    #calculate the forward pass
    # don't normalize the columns, normalize the ROWS (for LAYERNORM)
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1,keepdim=True) # Batch variance
    
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit var
    self.out = self.gamma * xhat + self.beta
    
    # update buffers NOT NECESSARY FOR LAYERNORM
    
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class Head(nn.Module):
  """ one head of self-attention"""

  def __init__(self, head_size):
    super().__init__()

    # create key query value in linear layers
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)

    # create 'tril'  (not a variable of module, but assigned using a register buffer)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


  def forward(self,x):
    B,T,C, = x.shape
    k = self.key(x) # B, T, 32
    q = self.query(x) # B, T, 32

    # compute attention scores "affinities"
    wei = q @ k.transpose(-2, -1) * C**-0.5 # B, T, C @ B,C, T -> B, T, T
    wei = wei.masked_fill(self.tril == 0, float('-inf'))
    wei = F.softmax(wei, dim = -1)

    # it's a linear nn -> makes so the output is 4,8,16 or B,T, head_size
    # x is 'private' to the token
    v = self.value(x)
    out = wei @ v
    return out


# implement the multi-head attention block (concatenating multiple single-head attention)
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    # introducing projection for residual block (linear projection of outcome of the concatenated layer)
    self.proj = nn.Linear(num_heads * head_size, n_embd)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenating over the channel dimension
    out = self.proj(out) # we apply the projection to get the result
    return out


# feed forward to improve the calculation of the logits
class FeedForward(nn.Module):
  """ a simple linear layer followed by non-linearity (relu)"""

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      # inner layer should be multiplied by 4 (from the attention peper)
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), 
    )

  # applies Linear on the per-token level (think on the data individually after self-attending)
  def forward(self, x):
    return self.net(x)


class Block(nn.Module):
  """ transformer block: communication FOLLOWED by computation"""
  
  def __init__(self, n_embd, n_head):

    # n_embd: embedding dimensions; n_head: the # of heads we want
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    # per token normalization that happens at the batch dimension
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # "x + " in block applies the residual connections "forks off and does computation, comes back"
    x = x + self.ffwd(self.ln2(x))
    return x

# we want to create a level of inderection (no embedding for logits, but an intermediate phase)
class BigramLanguageModel(nn.Module):
  # NOTE: removed vocab_size from the constructor
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    # n_embd is the number of embedding dimensions (suggested 32)
    # to go from token embeddings to logits we're going to add a linear layer

    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # encode on the position of the tocken as well
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    
    # interspersing communication and feedforward many times
    self.blocks = nn.Sequential(
      Block(n_embd, n_head = 4),
      Block(n_embd, n_head = 4),
      Block(n_embd, n_head = 4),
      nn.LayerNorm(n_embd),

    )
    self.lm_head = nn.Linear(n_embd, vocab_size)

    

  # feed forward 
  def forward(self,idx, targets=None):
    B, T = idx.shape
    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B, T, C) batch by time by channel tensor
    pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # creates (T,C)
    x = tok_emb + pos_emb # B,T,C
    x = self.blocks(x) # feed into self-attention head
    logits = self.lm_head(x)  # (B, T, vocab_size) -> we say that vocab_size is eq to C
    # interprets as logits which is score for next char in sequence
    # predicting what comes next from tokens in table


    if targets is None:
      loss = None
    else:
      # predicting the loss, we should see logit of correct has higher prob
      # cross_entropy wants (B C T) not (B T C) so reshape

      B, T, C = logits.shape
      logits = logits.view(B*T, C) # 2 dim array
      targets = targets.view(B*T) # 1 dimensional array
      loss = F.cross_entropy(logits, targets) 
    return logits, loss
      # output is the batch size by context size by all characters in vocab set
  

  # history eventually WILL be used, so we're allowing longer contexts
  def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):

      # crop idx to the last block_size tokens
      # because we're using positional embeddings, we can never have more than block_size going in
      idx_cond = idx[:, -block_size:]

      # get the predictions
      logits, loss = self(idx_cond)

      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B,C)

      # apply softmax to get probs
      probs = F.softmax(logits, dim=-1) # B,C

      # sample from teh distribution
      idx_next = torch.multinomial(probs, num_samples=1) #B, 1

      # append sampled inddex to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) #B, T+!

    return idx

model = BigramLanguageModel()
m = model.to(device)

# create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr =1e-3)
batch_size = 32

begin = datetime.datetime.now()
for iter in range(max_iters):

  if iter % eval_interval == 0:
    losses = estimate_loss()
    diff = str(datetime.datetime.now() - begin)
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} --- time elapsed: {diff}")

  # sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print('done')
context = torch.zeros((1,8), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) 