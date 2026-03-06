#flash attention raw kernel

import cupy as cp

flash_attention_kernel_code = r'''
extern "C" __global__
void flash_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N,
    int d,
    float scale
)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int Bc = blockDim.x;

    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + Bc * d;

    float o = 0.0f;
    float m = -1e9f;
    float l = 0.0f;

    const float* q = Q + row * d;
    int num_tiles = (N + Bc - 1) / Bc;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * Bc;
        int global_idx = tile_start + tid;

        if (global_idx < N) {
            for (int j = 0; j < d; j++) {
                K_tile[tid * d + j] = K[global_idx * d + j];
                V_tile[tid * d + j] = V[global_idx * d + j];
            }
        } else {
            for (int j = 0; j < d; j++) {
                K_tile[tid * d + j] = 0.0f;
                V_tile[tid * d + j] = 0.0f;
            }
        }
        __syncthreads();

        float m_new = m;
        float s[64];

        for (int i = 0; i < Bc; i++) {
            if (tile_start + i < N) {
                float dot = 0.0f;
                for (int j = 0; j < d; j++) {
                    dot += q[j] * K_tile[i * d + j];
                }
                s[i] = dot * scale;
                m_new = fmaxf(m_new, s[i]);
            } else {
                s[i] = -1e9f;
            }
        }

        float l_new = expf(m - m_new) * l;
        for (int i = 0; i < Bc; i++) {
            if (tile_start + i < N) {
                l_new += expf(s[i] - m_new);
            }
        }

        float correction = expf(m - m_new);
        o *= correction;
        for (int i = 0; i < Bc; i++) {
            if (tile_start + i < N) {
                float p = expf(s[i] - m_new);
                o += p * V_tile[i * d + tid];
            }
        }

        m = m_new;
        l = l_new;
        __syncthreads();
    }
    O[row * d + tid] = o / l;
}
'''

kernel = cp.RawKernel(flash_attention_kernel_code, "flash_attention")

def flash_attention(Q, K, V):
    Q = cp.ascontiguousarray(Q)
    K = cp.ascontiguousarray(K)
    V = cp.ascontiguousarray(V)
    N, d = Q.shape
    O = cp.zeros((N, d), dtype=cp.float32)
    scale = cp.float32(1.0 / (d ** 0.5))
    Bc = d
    shared_mem = 2 * Bc * d * 4
    kernel(
        (N,),
        (Bc,),
        (Q, K, V, O, cp.int32(N), cp.int32(d), scale),
        shared_mem=shared_mem
    )

    S = (Q @ K.T) * float(scale)
    m = cp.max(S, axis=-1, keepdims=True)
    l = cp.sum(cp.exp(S - m), axis=-1, keepdims=True)

    return O, l, m
