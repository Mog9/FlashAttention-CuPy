import cupy as cp
import numpy as np
import torch
from flash import flash_attention
from naive import naive_attention
from pure_backward import flash_attention_backward

def verify_backward():
    N, d = 512, 32

    Q_np = np.random.randn(N, d).astype(np.float32)
    K_np = np.random.randn(N, d).astype(np.float32)
    V_np = np.random.randn(N, d).astype(np.float32)
    dO_np = np.random.randn(N, d).astype(np.float32)

    Q_cp = cp.array(Q_np)
    K_cp = cp.array(K_np)
    V_cp = cp.array(V_np)
    dO_cp = cp.array(dO_np)

    _, l_cp, m_cp = naive_attention(Q_cp, K_cp, V_cp)
    O_cp = flash_attention(Q_cp, K_cp, V_cp)

    dQ_cp, dK_cp, dV_cp = flash_attention_backward(Q_cp, K_cp, V_cp, O_cp, dO_cp, l_cp, m_cp)

    Q_pt = torch.tensor(Q_np, requires_grad=True)
    K_pt = torch.tensor(K_np, requires_grad=True)
    V_pt = torch.tensor(V_np, requires_grad=True)
    dO_pt = torch.tensor(dO_np)

    scale = 1.0 / (d ** 0.5)
    S_pt = (Q_pt @ K_pt.T) * scale
    P_pt = torch.softmax(S_pt, dim=-1)
    O_pt = P_pt @ V_pt
    O_pt.backward(dO_pt)

    dQ_diff = float(cp.max(cp.abs(dQ_cp - cp.array(Q_pt.grad.numpy()))))
    dK_diff = float(cp.max(cp.abs(dK_cp - cp.array(K_pt.grad.numpy()))))
    dV_diff = float(cp.max(cp.abs(dV_cp - cp.array(V_pt.grad.numpy()))))

    print(f"dQ max diff: {dQ_diff:.6f}")
    print(f"dK max diff: {dK_diff:.6f}")
    print(f"dV max diff: {dV_diff:.6f}")

if __name__ == "__main__":
    verify_backward()