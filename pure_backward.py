import cupy as cp


def flash_attention_backward(Q, K, V, O, dO, l, m):
    d = Q.shape[-1]
    scale = 1.0 / (d ** 0.5)

    S = (Q @ K.T) *  scale #recompute
    P = cp.exp(S - m) / l

    dV = P.T @ dO #gradient through O = p @ v
    dP = dO @ V.T

    dS = P * (dP - cp.sum(dP * P, axis=-1, keepdims=True)) #softmax backward 
    dS = dS * scale

    dQ = dS @ K #S = Q @ K.T
    dK = dS.T @ Q

    return dQ, dK, dV