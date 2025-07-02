from xformers.ops import memory_efficient_attention
import torch

def xformers_attention(Q, K, V):
    """
    使用xFormers实现的高效Attention计算。
    Q: (B, H, L, D)
    K: (B, H, L, D)
    V: (B, H, L, D)
    返回: (B, H, L, D)
    """
    # xformers要求输入为(B, H, L, D)，与本项目一致
    return memory_efficient_attention(Q, K, V)
