import os, sys, datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameter setting
batch_size = 32 # how many independent sequences in parallel?
block_size = 8 # maximum context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # just in case we have cuda
eval_iters = 200


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


# we want to create a level of inderection (no embedding for logits, but an intermediate phase)
class BigramLanguageModel(nn.Module):
  # NOTE: removed vocab_size from the constructor
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    # n_embd is the number of embedding dimensions (suggested 32)
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # to go from token embeddings to logits we're going to add a linear layer
    self.lm_head = nn.linear(n_embd, vocab_size)

  # feed forward 
  def forward(self,idx, targets=None):
    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B, T, C) batch by time by channel tensor
    logits = self.lm_head(tok_emb)
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
      # get the predictions
      logits, loss = self(idx)

      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B,C)

      # apply softmax to get probs
      probs = F.softmax(logits, dim=-1) # B,C

      # sample from teh distribution
      idx_next = torch.multinomial(probs, num_samples=1) #B, 1

      # append sampled inddex to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) #B, T+!

    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr =1e-3)
batch_size = 32


for iter in range(max_iters):

  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].to_list()))