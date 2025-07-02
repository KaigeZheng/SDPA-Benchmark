import torch

def torch_attention(Q, K, V):
    """
    使用torch的scaled_dot_product_attention实现的Attention。
    Q: (B, H, L, D)
    K: (B, H, L, D)
    V: (B, H, L, D)
    返回: (B, H, L, D)
    """
    # PyTorch的SDPA要求输入为(B, H, L, D)
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    return output
