# standard attention, pure cupy ops
import cupy as cp

def naive_attention(Q, K, V):
    d = Q.shape[-1]
    S = (Q @ K.T) / cp.sqrt(d) #scale
    m = cp.max(S, axis=-1, keepdims=True)
    S = S- m
    exp_S = cp.exp(S)
    l = cp.sum(exp_S, axis=-1, keepdims=True)
    P = exp_S / l
    O = P @ V
    return O, l, m
