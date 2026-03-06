import cupy as cp
import torch
from naive import naive_attention
from flash import flash_attention
from pure_backward import flash_attention_backward
from benchmark import benchmark, benchmark_torch
from plot import plot
cp.set_printoptions(threshold=10)

def main():
    N, d = 512, 32
    Q = cp.random.randn(N, d, dtype=cp.float32)
    K = cp.random.randn(N, d, dtype=cp.float32)
    V = cp.random.randn(N, d, dtype=cp.float32)
    dO = cp.random.randn(N, d, dtype=cp.float32)

    out_naive, l, m = naive_attention(Q, K, V)
    out_flash = flash_attention(Q, K, V)

    dQ, dK, dV = flash_attention_backward(Q, K, V, out_flash, dO, l, m)

    Q_pt = torch.tensor(cp.asnumpy(Q), requires_grad=True)
    K_pt = torch.tensor(cp.asnumpy(K), requires_grad=True)
    V_pt = torch.tensor(cp.asnumpy(V), requires_grad=True)
    
    dO_pt = torch.tensor(cp.asnumpy(dO))
    scale = 1.0 / (d ** 0.5)
    O_pt = torch.softmax((Q_pt @ K_pt.T) * scale, dim=-1) @ V_pt
    O_pt.backward(dO_pt)

    def naive_fwd(Q, K, V):
        out, _, _ = naive_attention(Q, K, V)
        return out

    Q_pt_b = torch.tensor(cp.asnumpy(Q), device="cuda")
    K_pt_b = torch.tensor(cp.asnumpy(K), device="cuda")
    V_pt_b = torch.tensor(cp.asnumpy(V), device="cuda")

    def torch_fwd(Q, K, V):
        scale = 1.0 / (d ** 0.5)
        return torch.softmax((Q_pt_b @ K_pt_b.T) * scale, dim=-1) @ V_pt_b

    time_naive = benchmark(naive_fwd, Q, K, V)
    time_flash = benchmark(flash_attention, Q, K, V)
    time_torch = benchmark_torch(torch_fwd, Q_pt_b, K_pt_b, V_pt_b)

    speedup = time_naive / time_flash

    print(f"\n{'':30s} {'Naive':>12} {'Flash':>12} {'PyTorch':>12}")
    print(f"{'-'*66}")
    print(f"{'Time (ms)':30s} {time_naive:>12.3f} {time_flash:>12.3f} {time_torch:>12.3f}")
    print(f"{'Speedup vs Naive':30s} {'1.00x':>12} {speedup:>11.2f}x {time_naive/time_torch:>11.2f}x")

    plot(time_naive, time_flash, time_torch)

if __name__ == "__main__":
    main()