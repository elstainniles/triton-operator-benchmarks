"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(d):
    assert d['g'] is None
    B, T, H, K = d['k'].shape
    S = d['chunk_size']
    HV = d['beta'].shape[-1]
    assert S == 64 and T % S == 0 and (HV == H)
    k = d['k'].double().reshape(B, T // S, S, H, K).permute(0, 1, 3, 2, 4)
    beta = d['beta'].double().reshape(B, T // S, S, HV).permute(0, 1, 3, 2)
    lower = torch.tril(k @ k.transpose(-1, -2) * beta[..., None], diagonal=-1)
    eye = torch.eye(S, dtype=torch.float64).expand_as(lower)
    result = torch.linalg.solve_triangular(lower + eye, eye, upper=False, unitriangular=True)
    alt = torch.zeros_like(result)
    for i in range(S):
        alt[..., i, i] = 1
        if i:
            alt[..., i, :i] = -(lower[..., i, :i, None] * alt[..., :i, :i]).sum(-2)
    torch.testing.assert_close(result, alt, rtol=1e-10, atol=1e-10)
    residual = ((lower + eye) @ result - eye).abs().max().item()
    return (result.permute(0, 1, 3, 2, 4).reshape(B, T, HV, S).to(d['k'].dtype), dict(solve_vs_recurrence_max_abs=(result - alt).abs().max().item(), inverse_residual_max_abs=residual))
