import torch
import triton
import triton.language as tl

# Triton kernel for attention forward
@triton.jit
def attention_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    B, H, M, N, D,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)
    b = pid_hz // H
    h = pid_hz % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    # pointers
    Q_ptr = Q + b * stride_qz + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    K_ptr = K + b * stride_kz + h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    V_ptr = V + b * stride_vz + h * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    # load Q: [BLOCK_M, BLOCK_D]
    q = tl.load(Q_ptr, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D), other=0.0)
    k = tl.load(K_ptr, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
    v = tl.load(V_ptr, mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
    # 统一类型为float32进行softmax和概率计算
    q = q.to(tl.float32)
    k = k.to(tl.float32)
    v = v.to(tl.float32)
    # compute QK^T: [BLOCK_M, BLOCK_N]
    qk = tl.dot(q, tl.trans(k)) * sm_scale
    # softmax
    qk_max = tl.max(qk, 1)
    qk = qk - qk_max[:, None]
    qk_exp = tl.exp(qk)
    qk_exp_sum = tl.sum(qk_exp, 1)
    p = qk_exp / qk_exp_sum[:, None]
    # output: [BLOCK_M, BLOCK_D]
    out = tl.dot(p, v)
    # 转回原始dtype
    out = out.to(Q.dtype.element_ty)
    # store
    Out_ptr = Out + b * stride_oz + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(Out_ptr, out, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D))

def triton_attention(Q, K, V):
    """
    Q, K, V: [B, H, M, D], [B, H, N, D], [B, H, N, D]
    return: [B, H, M, D]
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    B, H, M, D = Q.shape
    N = K.shape[2]
    assert K.shape == (B, H, N, D) and V.shape == (B, H, N, D)
    dtype = Q.dtype
    Out = torch.empty((B, H, M, D), device=Q.device, dtype=dtype)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, D)
    sm_scale = 1.0 / (D ** 0.5)
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    attention_fwd_kernel[grid](
        Q, K, V, Out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        B, H, M, N, D,
        sm_scale,
        BLOCK_M, BLOCK_N, BLOCK_D,
        num_warps=4,
        num_stages=2
    )
    return Out
