import torch
import time
import numpy as np
from attention.attention_baseline import baseline_attention
from attention.attention_torch import torch_attention
from attention.attention_xformers import xformers_attention
from attention.attention_triton import triton_attention
from attention.attention_cublas import cublas_attention
import platform
import psutil

# from attention_cuda import cuda_attention_fp32, cuda_attention_fp16

def benchmark_attention(fn, name, seq_lens, loop_num=100, warm_up=5, B=1, H=8, D=64, dtype=torch.float32, device=torch.device("cuda")):
    print(f"\n{name} ({dtype})")
    for L in seq_lens:
        Q = torch.randn(B, H, L, D, device=device, dtype=dtype)
        K = torch.randn(B, H, L, D, device=device, dtype=dtype)
        V = torch.randn(B, H, L, D, device=device, dtype=dtype)
        
        # Warm-up
        for _ in range(warm_up):
            fn(Q, K, V)
        torch.cuda.synchronize()
        
        # Performance Test
        start = time.time()
        for _ in range(loop_num):
            fn(Q, K, V)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        tokens = B * L * loop_num
        speed = tokens / elapsed
        flops = 4 * B * H * (L ** 2) * D * loop_num
        tflops = flops / (elapsed * 1e12)
        print(f"  SeqLen={L:<5}  Speed={speed:>12.2f} tokens/s  TFLOPS={tflops:>7.2f}")

if __name__ == "__main__":
    # Print CPU info
    print("===== CPU Information =====")
    print(f"CPU Model: {platform.processor()}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}  Threads: {psutil.cpu_count(logical=True)}")
    print(f"Total Memory: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
    print(f"System: {platform.system()} {platform.release()} ({platform.version()})")
    print()
    # Print GPU info
    if torch.cuda.is_available():
        print("===== GPU Information =====")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)} GB")
            print(f"  CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        print()
    else:
        print("No available GPU device detected!\n")

    seq_lens = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    benchmark_attention(baseline_attention, "Baseline (cpu)", seq_lens, device=torch.device("cpu"))
    benchmark_attention(baseline_attention, "Baseline (cpu) FP16", seq_lens, device=torch.device("cpu"), dtype=torch.float16)
    benchmark_attention(baseline_attention, "Baseline (gpu)", seq_lens)
    benchmark_attention(baseline_attention, "Baseline (gpu) FP32", seq_lens, dtype=torch.float16)
    benchmark_attention(cublas_attention, "cuBLAS (cupy)", seq_lens)
    benchmark_attention(cublas_attention, "cuBLAS (cupy) FP16", seq_lens, dtype=torch.float16)
    benchmark_attention(torch_attention, "Torch SDPA", seq_lens)
    benchmark_attention(torch_attention, "Torch SDPA FP16", seq_lens, dtype=torch.float16)
    benchmark_attention(xformers_attention, "XFormers", seq_lens)
    benchmark_attention(xformers_attention, "XFormers FP16", seq_lens, dtype=torch.float16)
    benchmark_attention(triton_attention, "Triton", seq_lens)
    benchmark_attention(triton_attention, "Triton FP16", seq_lens, dtype=torch.float16)