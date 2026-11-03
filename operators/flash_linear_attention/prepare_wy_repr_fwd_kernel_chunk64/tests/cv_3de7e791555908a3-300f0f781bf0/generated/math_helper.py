"""Existing independent CPU equations; file-generation main omitted."""
import torch

def errors(a, b):
    e = (a - b).abs()
    return dict(max_abs=float(e.max()), rms=float(e.square().mean().sqrt()))

def wy_math(d):
    a = d['A_ab']
    B, T, H, BT = a.shape
    assert d['chunk_size'] == 64 and BT == 64 and (T % BT == 0) and (d['cu_seqlens'] is None)
    blocks = a.double().reshape(B, T // BT, BT, H, BT).permute(0, 1, 3, 2, 4).contiguous()
    lower = blocks.tril(-1)
    eye = torch.eye(BT, dtype=torch.float64).expand_as(lower)
    system = eye - lower
    inv = torch.linalg.solve_triangular(system, eye, upper=False, unitriangular=True)
    series = eye.clone()
    power = eye.clone()
    for _ in range(1, BT):
        power = power @ lower
        series += power
    residual = system @ inv - eye
    assert torch.allclose(inv, series, atol=1e-10, rtol=1e-10)
    assert residual.abs().max() < 1e-10
    result = inv.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT).contiguous()
    return ({'A_ab_inv': result}, dict(triangular_solve_vs_finite_neumann_series=errors(inv, series), system_residual_max_abs=float(residual.abs().max()), strict_lower_input=True, upper_and_diagonal_ignored=True))
