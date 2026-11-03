"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def reference_kkt(data):
    k, g, beta = (data['k'], data['g'], data['beta'])
    B, T, H, K = k.shape
    HV, BT = (data['HV'], data['BT'])
    result = torch.zeros_like(data['A'])
    scalar_errors = []
    for b in range(B):
        for h in range(HV):
            for start in range(0, T, BT):
                n = min(BT, T - start)
                keys = k[b, start:start + n, h // (HV // H)].double()
                dots = keys @ keys.T
                if g is not None:
                    gates = g[b, start:start + n, h]
                    differences = (gates[:, None] - gates[None, :]).float().double()
                    dots = dots * torch.exp(differences)
                dots = dots * beta[b, start:start + n, h, None].double()
                expected = torch.tril(dots, diagonal=-1)
                result[b, start:start + n, h, :n] = expected.to(result.dtype)
                if b == 0 and h == 0 and (start == 0):
                    for i, j in [(1, 0), (n - 1, 0), (n - 1, n - 2)]:
                        scalar = math.fsum((float(keys[i, d]) * float(keys[j, d]) for d in range(K)))
                        if g is not None:
                            scalar *= math.exp(float(differences[i, j]))
                        scalar *= float(beta[b, start + i, h])
                        scalar_errors.append(abs(float(expected[i, j]) - scalar))
    assert max(scalar_errors) < 1e-10 and torch.isfinite(result).all()
    return ({'A': result}, {'scalar_max_abs_difference': max(scalar_errors), 'method': 'CPU float64 K K^T, original gate-operand cast, row beta and strictly lower triangular mask', 'original_acceptance': {'rtol': 0.001, 'atol': 0.001, 'equal_nan': True}})
