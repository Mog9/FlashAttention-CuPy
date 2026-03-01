# standard attention, pure cupy ops
import cupy as cp

def naive_attention(Q, K, V):
    scores = Q @ K.T #dot
    d = Q.shape[-1]
    scores = scores / cp.sqrt(d) #scale
    
    #numerically stable softmax
    scores = scores - cp.max(scores, axis=-1, keepdims=True)
    scores = cp.exp(scores)
    scores = scores / cp.sum(scores, axis=-1, keepdims=True)

    return scores @ V
