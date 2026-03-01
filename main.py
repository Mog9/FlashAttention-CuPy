import cupy as cp
from naive import naive_attention
from flash import flash_attention
from benchmark import benchmark
from plot import plot

def main():
    N, d = 512, 32
    Q = cp.random.randn(N, d, dtype=cp.float32)
    K = cp.random.randn(N, d, dtype=cp.float32)
    V = cp.random.randn(N, d, dtype=cp.float32)

    out_naive = naive_attention(Q, K, V)
    out_flash = flash_attention(Q, K, V)
    diff = float(cp.max(cp.abs(out_naive - out_flash)))
    print(f"correctness check - max diff: {diff:.6f}")

    time_naive = benchmark(naive_attention, Q, K, V)
    time_flash = benchmark(flash_attention, Q, K, V)
    speedup = time_naive / time_flash

    print(f"\n{'':30s} {'Naive':>15} {'Flash':>15}")
    print(f"{'-'*60}")
    print(f"{'Time (ms)':30s} {time_naive:>15.3f} {time_flash:>15.3f}")
    print(f"{'Speedup':30s} {'1.00x':>15} {speedup:>14.2f}x")

    plot(time_naive, time_flash)

if __name__ == "__main__":
    main()