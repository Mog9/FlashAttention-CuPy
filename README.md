# Flash Attention from Scratch in CUDA / CuPy

A from-scratch implementation of Flash Attention with a full forward and backward pass. Forward is a custom CUDA kernel written in CuPy RawKernel. Backward is implemented in pure CuPy ops. Both are verified correct against PyTorch.

---

## What is Flash Attention

Standard attention computes the full NxN score matrix and writes it to global memory. For long sequences this becomes the bottleneck — not compute, but memory bandwidth.

Flash Attention processes attention in tiles. The NxN score matrix is never materialized in HBM. Scores are computed tile by tile in SRAM using online softmax, then discarded. Memory complexity goes from O(N²) to O(N).

---

## Forward Pass

**Algorithm:**

Outer loop — iterate over tiles of Q

Inner loop — for each Q tile, iterate over all K and V tiles

For each tile:
- Compute score tile S = Q_tile @ K_tile.T / sqrt(d) in registers
- Update running max m and running sum l using online softmax
- Apply correction factor exp(m_old - m_new) to rescale prior accumulation
- Accumulate O += exp(S - m_new) @ V_tile

After all tiles — divide O by l, write O tile to HBM once

**Online softmax** is the core insight — exact softmax computed incrementally without seeing all scores at once, using a running maximum and denominator corrected each tile.

The forward kernel is a CuPy RawKernel. Each thread handles one output dimension. All threads in a block collaborate to load K and V tiles into shared memory, then each thread accumulates its own output element independently.

---

## Backward Pass

**Recomputation trick** — standard attention stores the full NxN softmax weight matrix P from forward to use during backward, costing O(N²) memory. Flash attention backward throws P away and recomputes it from Q, K, and the saved scalars m and l. Same result, O(N) memory.

**Algorithm:**

```
# recompute attention weights
S = Q @ K.T / sqrt(d)
P = exp(S - m) / l

# gradient through O = P @ V
dV = P.T @ dO
dP = dO @ V.T

# softmax backward
dS = P * (dP - rowsum(dP * P))
dS = dS / sqrt(d)

# gradient through S = Q @ K.T
dQ = dS @ K
dK = dS.T @ Q
```

The softmax backward formula `dS = P * (dP - rowsum(dP * P))` comes from the Jacobian of softmax. The rowsum term corrects for the coupling between softmax outputs — changing one score affects all probabilities in the row.

---

## Results

Tested on RTX 3050 (4GB), N=512, d=32, float32.

```
                               Naive CuPy       Flash Kernel        PyTorch
Time (ms)                           0.559              2.867          0.049
```

Flash kernel is slower than both naive CuPy and PyTorch. Naive CuPy uses cuBLAS under the hood. PyTorch uses cuDNN. Matching their performance requires warp-level score parallelization — scores need to be computed once per block into shared memory rather than redundantly per thread. That is the next step.

The kernel is mathematically correct. Gradients match PyTorch autograd to 1e-6.

---

## Correctness Verification

Forward — compared against naive CuPy attention, max diff under 1e-6.

Backward — dQ, dK, dV compared against PyTorch autograd on identical inputs, max diff under 1e-6.

---

## Project Structure

```
flash-attention/
├── naive.py           # standard attention in pure CuPy, returns O, l, m
├── flash.py           # Flash Attention forward RawKernel
├── backward.py        # Flash Attention backward in pure CuPy
├── benchmark.py       # CUDA event timing for CuPy and PyTorch
├── torch_verify.py    # backward correctness check against PyTorch
├── plot.py            # bar plot visualization
├── main.py            # entry point, runs all checks and benchmarks
└── README.md
```

---

## Setup

```bash
pip install cupy-cuda12x matplotlib torch
```

Requires CUDA 12.x and a compatible NVIDIA GPU.

---

## Run

```bash
python main.py
```

Runs forward correctness check, backward correctness check against PyTorch, prints timing comparison, saves plot to `flash_results.png`.

---

Online softmax lets you compute exact softmax tile by tile without storing the full score row. The recomputation trick trades compute for memory in the backward pass — recomputing P on the fly instead of storing it keeps memory O(N) through the entire forward and backward. The gap between a correct kernel and a fast one is large — closing it requires warp-level parallelism and careful thread coordination, which is what makes production Flash Attention implementations genuinely hard.