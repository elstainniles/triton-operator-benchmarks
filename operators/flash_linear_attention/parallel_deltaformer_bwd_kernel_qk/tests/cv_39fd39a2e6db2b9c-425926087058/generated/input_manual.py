"""Existing independent CPU equations; file-generation main omitted."""
import torch

def build(kind, src):
    seed = 42
    gen = torch.Generator().manual_seed(seed)

    def r(shape, dtype=torch.float32):
        return (torch.randn(shape, generator=gen) * 0.1).to(dtype)
    import math
    T, H, D = (1024, 4, 128)
    q = torch.randn(T, H, D, generator=gen, dtype=torch.float16)
    k = torch.randn(T, H, D, generator=gen, dtype=torch.float16)
    u = torch.randn(T, H, D, generator=gen, dtype=torch.float16)
    grad_v = torch.randn(T, H, D, generator=gen, dtype=torch.float16)
    beta = torch.randn(T, H, generator=gen, dtype=torch.float16).sigmoid()
    qk_scale = D ** (-0.5)
    fa_scale = qk_scale / math.log(2)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    lse = torch.zeros(T, H)
    row_dot_sum = torch.zeros(T, H)
    for h in range(H):
        scores = q[:, h].float() @ k[:, h].float().T * fa_scale
        scores = scores.masked_fill(~mask, float('-inf'))
        rowmax = scores.max(-1).values
        weights = torch.exp2(scores - rowmax[:, None])
        lse[:, h] = rowmax + torch.log2(weights.sum(-1))
        p = torch.exp2(scores - lse[:, h, None])
        dp = grad_v[:, h].float() @ u[:, h].float().T
        row_dot_sum[:, h] = (p * dp).sum(-1)
    return (dict(input_q=q, input_k=k, input_u=u, input_lse=lse, input_grad_v=grad_v, input_beta=beta, input_row_dot_sum=row_dot_sum, params={'H': H, 'T': T, 'D': D, 'fa_scale': fa_scale, 'qk_scale': qk_scale}), seed)
