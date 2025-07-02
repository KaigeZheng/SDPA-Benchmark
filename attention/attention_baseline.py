import torch

def baseline_attention(Q, K, V):
    """
    最普通的PyTorch实现的Attention。
    Q: (B, H, L, D)
    K: (B, H, L, D)
    V: (B, H, L, D)
    返回: (B, H, L, D)
    """
    # Q @ K^T
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output 