from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# --------------------------------------------------------------------------------------




class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # for scaling weights by 1/sqrt(N) - flag for this module
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a 'bias', more of a mask but following the openAI/HF naming convention
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B, T, C = x.size() # batch size, sequence length, embeddign dimensionality (n_embd)
        # calculate query, key values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is the "head size", and C (number of channels) = nh * hs
        # e.g. in gpt2 (124m), n_head = 12, hs = 64 so nh*hs = C = 768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # doing a bunch of tensor gymnastics to make this work as a single module
        # where previously we had a single head and concatenated to multihead

        # description: 
        # each head has sequences of tokens (1024 in this model)
        # token emits 3 vectors: query, key, and value
        # queries and keys have to multiply each other to get the "attention amount"
            # mutliplicative interaction
        # how this works is we make the # heads into a batch dimension just like B
        # treats B and nh like batches and just applies everything at once

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #interaction of queries and keys
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) #autoregressive mask -> no future looking
        att = F.softmax(att, dim=-1) # normalizes attention so it sums to 1
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) # weighted sum of values that we found interesting
        y = y.transpose(1, 2).contiguous().view(B, T, C) #re-assemble all head outputs side by side
        # this last step actually concatenates the full setup
        # it's equivalent mathematically to our previous setup

        # output projection
        y = self.c_proj(y)
        return(y)


# "kernel fusion" helps speed up the "round trips" from GPU memory through GPU, CPU, memory to do operations
# by doing everything at once
#class TanhGELU(nn.Module):
#    def forward(self, input):
#        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd) # why 4?
        self.gelu   = nn.GELU(approximate='tanh') # activation function
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # project back out
        # for scaling weights by 1/sqrt(N) - flag for this module
        self.c_proj.NANOGPT_SCALE_INIT = 1
    # Note -> GELU = cumulative dist for gaussian distribution
    # approx gelu is 'tanh' estimated a little faster but NOW it's just as fast (this is a historical quirk)
    # there are some reasonings that make this work better for optimization
    # GELU is better than ReLU because it never forces a gradient node to be zeroed out forever
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



# now let's do the block which is in the self.transformer ModuleDict
class Block(nn.Module):

    # note that the layer norms are BEFORE the attention head and MLP'
    # which deviates from the og AIAYN paper
    # you want a clean residual stream from the beginning to end
        # addition just distributes gradienst (from micrograds) equally
        # so the gradients from top flow straight through to the inputs unchanged
        # in addition, the gradients flow through the blocks which change the optim over time
        # a cleaner pathway is desirable from an optimization perspective
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) # we will probaly define this later
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    #forward pass of what the block computes
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # aggregation/weighted sum fucntion (reduce)
        x = x + self.mlp(self.ln_2(x)) # happens with every single token indivdiually (map)
        return x
    



# setting values to match GPT2-124M model params
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # 50k BPE merges, 256 byte tokens, 1 end of text token
    n_layer: int = 12 # num layers
    n_head: int = 12 # num heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
        # using an nn module dict that lets you index into submodules using keys (just like a dict)
        # h uses nn.ModuleList to define and index based on the number ( 0 to 11 )
        # additional final layernorm
        # final classifier -> language mdoel head which projects from the enbedding layer to the final amount
            # note -> uses no bias for the final layer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding weights
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding weights
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # hidden layer?
            ln_f = nn.LayerNorm(config.n_embd), # layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        # taking the wte.weight and redirecting it to point to the lm_head
        # copies the reference, the old value is orphaned & pytorch cleans that up
        # result is a single tensor used twice in the forward pass
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    # apply init weight function
    def _init_weights(self, module):
        std = 0.02
        # if they are an nn.Linear, we initialize the weight as a norm
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *=  (2 * self.config.n_layer) ** -0.5 # *2 because every layer has MLP and attention
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # if there is a bias, we initialize to zero
            # this is a little diff -> usually it's initialized to a uniform dist
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        # only other layer initialized with params is LayerNorm, 
            # and this is initialized with scale of 1 and offset of 0 which is what we want
            # so we keep it as is
        # Normally, we would use 1/sqrt(d_model) as the variance, but this is close to 0.02


    def forward(self, idx, targets=None):
        # idx is of shape (B, T) -> batch and time, T can't be more than block size
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length of {T}, block_size is only {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb


        # overall note: when we forward, we're calling the module itself 
        # thus block(x) is self.transformer.h[block]
        # which calculates the forward pass -> x + self attn_(self.ln_1(x)) ..etc
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss




    # adding a method to pull in the pre-trained model
    @classmethod
    def from_pretrained(cls, model_type):
        """ loads pre-trained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrianed gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2":         dict(n_layer=12, n_head=12, n_embd=768), #124M params
            "gpt2-medium":  dict(n_layer=24, n_head=16, n_embd=1024), #350M params
            "gpt2-large":   dict(n_layer=36, n_head=20, n_embd=1280), #774M params
            "gpt2-xl":      dict(n_layer=48, n_head=25, n_embd=1600), #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer

        # init a huggingface/transformers model (basically what we're doing with this module)
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # we have to do a bunch of manipulation because a few operations are not totally aligned
        # copy while ensuring all of the params are aligned and match in names & shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore masked bias
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just bias
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        # ensure the keys match
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights that we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            
            else:
                # straight copy other params
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        
        return model


# --------------------------------------------------------------
num_return_sequences = 5
max_length = 30


import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('./data/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T) + 1 > len(self.tokens):
            self.current_position = 0

        return x, y

# --------------------------------------------------------------
import time # we want to time performance

# auto-detect the device that is avail for us
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
print(f"using device: {device}")
#device = 'cpu' # override for now

# manual seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# to compensate OOM errors, let's lower batch size
# pro tip, use base2 numbers to most efficiently allocate operations
train_loader = DataLoaderLite(B=8, T=1024)

# trying some fuckery with the datatype to get it to run faster
# at a slightly lower precision (may not work on my machine though)
# default is "highest" (float32), "high" uses tensorfloat32 or 2 bfloat16 numbers if approx fast mult alg are avail
# "medium" uses bfloat16 datatype
torch.set_float32_matmul_precision('high')
# this doesn't really do much for my particular machine
# but it's free and will run ~3x faster on enabled machines



#model = GPT.from_pretrained('gpt2')
# putting our OWN model to the test (untrianed it's totally random)
model = GPT(GPTConfig())
model.to(device) # move all the tensors to GPU

# Using torch.compile -> compiles code which adds up front time but can save in the long run
# speedup mainly comes from reducing python overhead and GPU read/writes
# model = torch.compile(model) # NOTE: only works on python 3.8-3.11, may not work on windows


# ADAMW is a good algo to use, pretty good vs stochastic GD
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# optimizer in 50 steps
for i in range(50):
    t0 = time.time()
    # get the next batch of tokens and move them to the device
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    
    # ALWAYS ZERO THE GRADIENT WHEN CALCING BACKPROP
    optimizer.zero_grad()

    # autocasting to a lower precision float
    # note - be careful when autocasting - only use on forward pass pieces
    # not2 - use bfloat16 instead of float16, you would need to scale the gradients otherwise
    # this makes a HUGE diff on my machine!
    with torch.autocast(device_type=device, dtype=torch.bfloat16): 
        logits, loss = model(x, y)

    # allows us to interact with the code
    # import code; code.interact(local=locals())
    # we see that our logits are of dtype float32 -> this precision is wasted on us!
    # depending on the graphics card, you can get a lot more TFLOPS if you decrease precision
    # and there's a speed that you can access this memory - most workloads are memory-bound

    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() # take time to work through work that was scheduled to run
    t1 = time.time()
    dt = (t1-t0)*1000 # ms time difference
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms")


# skipping sampling logic for now
import sys; sys.exit(0)


# put it into eval mode whenever we just want to use it
# might not do anything as is but we want to just ensure we don't mess anything up

model.eval() 


# generate! 
# set seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)

        #take the logits at the last position 
        # we get ALL the logits here but we only care about the last column's logits
        logits = logits[:, -1, :] # (B, vocab_size)

        #get the probabilities (is this necessary? - yes but only bc of the random sampling)
        probs = F.softmax(logits, dim=-1)

        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is 5, 50
        # this helps keep the model on track with avoiding super low probability tokens
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)

        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

        # append to the sequence of tokens (our original sequence)
        x = torch.cat((x, xcol), dim=1)

#print the rows
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

 # Lambda labs -> where this can be trained on the cloud
 # TENSOR NOTES
 # what is a tensor core? 
    # matrix multiplier node 4x4 multiply
    # any ops that require matrix mult is broken up into this
    # since most of the work is matrix multiplication (linear layers)
    # a lot of what we do is well-tuned with GPU cores
    # biggest matrix mult is the classifier at the top -> it dominates anythign else that happens in the net
    # for reference: NVIDIA A100 architecture white paper
    # for a TF32 versus FP32 -> the precision bits get dropped to 10 (from 23) which allows faster calcs
