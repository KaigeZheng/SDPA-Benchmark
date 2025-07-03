import torch
import torch.nn as nn
import time
from attention.attention_baseline import baseline_attention
from attention.attention_torch import torch_attention
from attention.attention_xformers import xformers_attention
from attention.attention_triton import triton_attention
from attention.attention_cublas import cublas_attention

class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_fn):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_fn = attn_fn

    def forward(self, x):
        B, L, D = x.shape
        h = self.num_heads
        d = self.head_dim
        Q = self.q_proj(x).view(B, h, L, d)
        K = self.k_proj(x).view(B, h, L, d)
        V = self.v_proj(x).view(B, h, L, d)
        attn_out = self.attn_fn(Q, K, V)
        attn_out = attn_out.reshape(B, L, D)
        return self.out_proj(attn_out)

class GPT3Block(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_fn, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CustomAttention(embed_dim, num_heads, attn_fn)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT3Model(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, attn_fn):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.blocks = nn.ModuleList([
            GPT3Block(embed_dim, num_heads, attn_fn) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx):
        # idx: (B, L)
        B, L = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :L, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def benchmark_gpt3(attn_fn, name, seq_len=1024, batch_size=1, embed_dim=768, num_heads=12, num_layers=12, vocab_size=50257, device='cuda', dtype=torch.float16, warmup=5, steps=20):
    model = GPT3Model(vocab_size, seq_len, embed_dim, num_heads, num_layers, attn_fn).to(device=device, dtype=dtype)
    model.eval()
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(idx)
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(steps):
            _ = model(idx)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
    tokens = batch_size * seq_len * steps
    print(f"{name}: {tokens/elapsed:.2f} tokens/s, {elapsed/steps*1000:.2f} ms/iter")

if __name__ == "__main__":
    print("Benchmarking GPT-3 structure with different attention implementations")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    attn_list = [
        (baseline_attention, "Baseline (cpu)", "cpu"),
        (baseline_attention, "Baseline (gpu)", device),
        (cublas_attention, "cuBLAS", device),
        (torch_attention, "Torch SDPA", device),
        (xformers_attention, "XFormers", device),
        (triton_attention, "Triton", device),
    ]
    seq_lens = [256, 1024, 4096]
    # FP32
    print("\n===== FP32 Benchmark =====")
    for seq_len in seq_lens:
        print(f"\n--- Sequence Length: {seq_len} ---")
        for attn_fn, name, attn_device in attn_list:
            dtype = torch.float32
            benchmark_gpt3(attn_fn, f"{name} FP32 (seq_len={seq_len})", seq_len=seq_len, device=attn_device, dtype=dtype)
    # FP16
    if device == 'cuda':
        print("\n===== FP16 Benchmark =====")
        for seq_len in seq_lens:
            print(f"\n--- Sequence Length: {seq_len} ---")
            for attn_fn, name, attn_device in attn_list:
                dtype = torch.float16
                benchmark_gpt3(attn_fn, f"{name} FP16 (seq_len={seq_len})", seq_len=seq_len, device=attn_device, dtype=dtype)