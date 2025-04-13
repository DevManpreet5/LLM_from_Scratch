import torch
import gradio as gr
import tiktoken
import pandas as pd
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

class multiheadv2(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, attention_head, boolbias):
        super().__init__()
        self.head_dim = d_out // attention_head
        self.d_out = d_out
        self.attention_head = attention_head
        self.W_query = nn.Linear(d_in, d_out, bias=boolbias)
        self.W_key = nn.Linear(d_in, d_out, bias=boolbias)
        self.W_value = nn.Linear(d_in, d_out, bias=boolbias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_token, d_out = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_token, self.attention_head, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_token, self.attention_head, self.head_dim).transpose(1, 2)
        values = values.view(b, num_token, self.attention_head, self.head_dim).transpose(1, 2)
        attn_score = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_token, :num_token]
        attn_score.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_score / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_token, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale_params = nn.Parameter(torch.ones(emb_dim))
        self.shift_params = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.scale_params + self.shift_params

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class feedforward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config['emb_dim'], config['emb_dim'] * 4),
            GELU(),
            nn.Linear(config['emb_dim'] * 4, config['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = multiheadv2(d_in=config['emb_dim'], d_out=config['emb_dim'], context_length=config['context_length'], dropout=config['drop_rate'], attention_head=config['n_heads'], boolbias=config['qkv_bias'])
        self.Layernorm1 = LayerNorm(config['emb_dim'])
        self.Layernorm2 = LayerNorm(config['emb_dim'])
        self.feedforw = feedforward(config)
        self.dropout = nn.Dropout(config['drop_rate'])

    def forward(self, x):
        skip = x
        x = self.Layernorm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + skip
        skip = x
        x = self.Layernorm2(x)
        x = self.feedforw(x)
        x = self.dropout(x)
        x = x + skip
        return x

class GPT_2(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], num_classes)

    def forward(self, inputidx):
        batch_size, seq = inputidx.shape
        tokens = self.token_emb(inputidx)
        pos_embeds = self.pos_emb(torch.arange(seq, device=inputidx.device))
        x = tokens + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x[:, -1])
        return logits

tokenizer = tiktoken.get_encoding("gpt2")
pad_token_id = tokenizer.eot_token

df_temp = pd.read_csv("train.csv")
label_mapping = dict(enumerate(df_temp["target"].astype("category").cat.categories))
num_classes = len(label_mapping)
inv_label_mapping = {v: k for k, v in label_mapping.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT_2(GPT_CONFIG_124M, num_classes)
model.load_state_dict(torch.load("biofinetuned_partialEpoch1.pth", map_location=device))
model.to(device)
model.eval()

def classify_review(text, max_length=128):
    input_ids = tokenizer.encode(text)[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
    predicted_label = torch.argmax(logits, dim=-1).item()
    return label_mapping[predicted_label]

iface = gr.Interface(
    fn=classify_review,
    inputs=gr.Textbox(label="Enter Medical Abstract / Review"),
    outputs=gr.Textbox(label="Predicted Category"),
    title="MedGPT",
    description="Fast biomedical text classifier trained on domain-specific corpus"
)

iface.launch()
