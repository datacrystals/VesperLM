"""
Muon optimizer (Keller Jordan's Muon, single-file version).

Applied to 2D hidden weights only; embeddings, the LM head, and all
1D/bias params stay on AdamW (Muon is designed for hidden layers).

Key idea: Newton-Schulz orthogonalize each momentum update, so all
singular directions of the update get comparable magnitude. Empirically
~1.3-2x better sample efficiency than AdamW on transformer pretraining.

FP32-only (no foreach fused path needed on a P40).
"""

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate orthogonalization of G via Newton-Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.device.type == "cuda" else G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                if p.ndim >= 2:
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.reshape(p.shape), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)
