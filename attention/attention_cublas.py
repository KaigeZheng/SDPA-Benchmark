import cupy as cp
import torch
import torch.nn.functional as F

def cublas_attention(Q, K, V):
    """
    优化后的Attention实现，QK^T用cupy加速，其余用PyTorch高效kernel。
    Q, K, V: (B, H, L, D) torch.Tensor，cuda上
    返回: (B, H, L, D) torch.Tensor，cuda上
    """
    assert Q.device.type == 'cuda' and K.device.type == 'cuda' and V.device.type == 'cuda'
    B, H, L, D = Q.shape
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    # 只用cupy做QK^T
    with cp.cuda.Device(Q.device.index):
        Q_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(Q))
        K_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(K))
        attn_scores = cp.matmul(Q_cp.reshape(-1, L, D), K_cp.reshape(-1, L, D).transpose(0, 2, 1))
        attn_scores = attn_scores.reshape(B, H, L, L)
        attn_scores_torch = torch.utils.dlpack.from_dlpack(attn_scores.toDlpack())
    # softmax及后续全部用PyTorch
    attn_scores_torch /= (D ** 0.5)
    attn_probs = F.softmax(attn_scores_torch, dim=-1)
    output = torch.matmul(attn_probs, V)
    return output