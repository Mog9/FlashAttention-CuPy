import cupy as cp

def benchmark(fn, Q, K, V, iters=20):
    for _ in range(5):
        fn(Q, K, V)
    cp.cuda.runtime.deviceSynchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        fn(Q, K, V)
    end.record()
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / iters


def benchmark_torch(fn, Q, K, V, iters=20):
    import torch
    for _ in range(5):
        fn(Q, K, V)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters