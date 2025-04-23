import torch
import torch.nn as nn

torch.manual_seed(123)

vocab = set()

for i in range(10):
    for j in range(10):
        vocab.add(i * j)

vocab = vocab.union({'x', '=', '<|bos|>', '<|eos|>', '<|pad_id|>'})
vocab = list(vocab)

itos = {i:el for i, el in enumerate(vocab)}
stoi = {el:i for i, el in enumerate(vocab)}

encode = lambda seq: [stoi[el] for el in seq]
decode = lambda tokens: [itos[t] for t in tokens]

CONF = {
    'vocab_size': len(vocab),
    'context_length': 5, # a * b = c - 5 elements
    'emb_dim': 2, # need more than 1 but minimum in order to easier understand
    'n_heads': 2,
    'n_layers': 2,
    'drop_rate': 0.01, # 1%
    'qkv_bias': False, # just matrix transformation (projection) to Q,K,V from X
}

def make_dataset():
    X = []
    Y = []
    for i in range(10):
        for j in range(10):
            X.append(encode(['<|bos|>', i, 'x', j, '=']))
            Y.append(encode([i, 'x', j, '=', i * j]))
    return torch.tensor(X), torch.tensor(Y)


X, Y = make_dataset()


class MultiHeadAttention(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.d_in = conf['emb_dim']
        self.d_out = conf['emb_dim']
        self.head_dim = self.d_out // conf['n_heads']
        self.W_query = nn.Linear(self.d_in, self.d_out, bias=conf['qkv_bias'])
        self.W_key = nn.Linear(self.d_in, self.d_out, bias=conf['qkv_bias'])
        self.W_value = nn.Linear(self.d_in, self.d_out)
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.dropout = nn.Dropout(conf['drop_rate'])
        self.register_buffer('mask',
            torch.triu(torch.ones(conf['context_length'], conf['context_length']), diagonal=1))

    def forward(self, x):
        batch_dim, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_dim, num_tokens, self.conf['n_heads'], self.head_dim)
        values = values.view(batch_dim, num_tokens, self.conf['n_heads'], self.head_dim)
        queries = queries.view(batch_dim, num_tokens, self.conf['n_heads'], self.head_dim)

        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_dim, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
    

class FeedForward(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.layers = nn.Sequential(
            nn.Linear(conf['emb_dim'], 4 * conf['emb_dim']),
            nn.GELU(),
            nn.Linear(4 * conf['emb_dim'], conf['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.mha = MultiHeadAttention(conf)
        self.ff = FeedForward(conf)
        self.norm1 = nn.LayerNorm(conf['emb_dim'])
        self.norm2 = nn.LayerNorm(conf['emb_dim'])
        self.drop_shortcut = nn.Dropout(conf['drop_rate'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.mha(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPT(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.tok_emb = nn.Embedding(conf['vocab_size'], conf['emb_dim'])
        self.pos_emb = nn.Embedding(conf['context_length'], conf['emb_dim'])
        self.drop_emb = nn.Dropout(conf['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(conf) for _ in range(conf['n_layers'])],
        )

        self.final_norm = nn.LayerNorm(conf['emb_dim'])
        self.out_head = nn.Linear(conf['emb_dim'], conf['vocab_size'], bias=False)

    def forward(self, x):
        x = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(self.conf['context_length']))
        x = x + pos
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x  # logits


llm = GPT(CONF)
optimizer = torch.optim.Adam(llm.parameters())


def train(steps):
    llm.train()
    for step in range(steps):
        optimizer.zero_grad()
        logits = llm(X)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), Y.flatten())
        loss.backward()
        optimizer.step()
        print(f'#{step} step trained, loss {loss}')


def generate(x, temperature=0.8, top_p=0.9, top_k=None):
    x = x.view(1, *x.shape)
    with torch.no_grad():
        logits = llm(x)

    logits = logits[:, -1, :]  # last element is interested only

    if top_k is not None:
        logits, _ = torch.topk(logits, top_k)

    if temperature > 0:      
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)        
    else:
        next_token = torch.argmax(logits, dim=-1)  

    return decode(next_token.flatten().tolist())[0]


def sample_top_p(probs, p):
    probs_sort, prob_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(prob_idx, -1, next_token) 
    return next_token

# train(100000)
# generate(X)