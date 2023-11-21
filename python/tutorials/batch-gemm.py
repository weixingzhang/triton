

import torch

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,

        B, M, N, K,

        stride_qm, stride_qk,  #
        stride_kb, stride_kn, stride_kk,  #
        stride_om, stride_ok,

        BLOCK_B: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,  #
):
    pid = tl.program_id(0)

    offs_b = tl.arange(0, BLOCK_B)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_q = pid * BLOCK_M + offs_m
    q_ptrs = q_ptr + (offs_q[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    q = tl.load(q_ptrs)

    k_ptrs = k_ptr + offs_b[:, None, None] * stride_kb + offs_n[None, None, :] * stride_kn + offs_k[None, :, None] * stride_kk
    v_ptrs = v_ptr + offs_b[:, None, None] * stride_kb + offs_k[None, None, :] * stride_kk + offs_n[None, :, None] * stride_kn

    acc = tl.zeros((BLOCK_B, BLOCK_M, BLOCK_K), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        qk = tl.zeros((BLOCK_B, BLOCK_M, BLOCK_N), dtype=tl.float32)

        k  = tl.load(k_ptrs)
        qk = tl.dot(q, k)
        qk = qk.to(tl.float16)

        v  = tl.load(v_ptrs)
        acc += tl.dot(qk, v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_kn
    o = tl.sum(acc, axis=0)

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, o)


def matmul(q, k, v):
    # Check constraints.
    assert q.shape[-1] == k.shape[-1], "Incompatible dimensions"
    assert q.is_contiguous(), "Matrix A must be contiguous"
    assert q.is_contiguous(), "Matrix B must be contiguous"
    M, K = q.shape
    B, N, K = k.shape
    # Allocates output.
    o = torch.empty((M, K), device=q.device, dtype=q.dtype)

    BLOCK_B = B
    BLOCK_M = 16
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), 1, )
    matmul_kernel[grid](
        q, k, v, o, #
        B, M, N, K, #
        q.stride(0), q.stride(1),  #
        k.stride(0), k.stride(1),  k.stride(2), #
        o.stride(0), o.stride(1),  #
        BLOCK_B,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K
    )
    return o


#  q:  [16, 64], k: [4, 512, 64], v: [4, 512, 64]
# qk:  [4, 16, 512]
# qkv: [4, 16, 64]
torch.manual_seed(0)
q = torch.randn((16, 64), device='cuda', dtype=torch.float16)
k = torch.randn((4, 512, 64), device='cuda', dtype=torch.float16)
v = torch.randn((4, 512, 64), device='cuda', dtype=torch.float16)


triton_output = matmul(q, k, v)
# torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
# print(f"torch_output={torch_output}")
# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")
