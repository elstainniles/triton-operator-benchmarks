"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def reference(d, source):
    tensors = {key: d[key].double() for key in ('q', 'k', 'v', 'g', 'h', 'do', 'dh', 'dA', 'dg')}
    scale, bt = (float(d['scale'].item()), int(d['chunk_size'].item()))
    values = naive_chunk_gsa_bwd_k_dqkvg(*(tensors[key] for key in ('q', 'k', 'v', 'g', 'h', 'do', 'dh', 'dA', 'dg')), scale, bt, dtype=torch.float64)
    outputs = dict(zip(('A', 'dq', 'dk', 'dv', 'dgv'), values))
    q, k, v, g, do, da, dg = (tensors[key][0, :bt, 0] for key in ('q', 'k', 'v', 'g', 'do', 'dA', 'dg'))
    h, dh = (tensors['h'][0, 0, 0], tensors['dh'][0, 0, 0])
    width, channels = (k.shape[-1], v.shape[-1])
    errors = []
    for row in (0, bt - 1):
        for col in (0, min(row, bt - 1)):
            expected = scale * math.fsum((float(q[row, j]) * float(k[col, j]) for j in range(width)))
            actual = float(outputs['A'][0, row, 0, col])
            errors.append(abs(expected - actual))
            assert abs(expected - actual) < 1e-10 * (1 + abs(expected))
        for dim in (0, width - 1):
            dq = scale * math.fsum((float(do[row, j]) * 2 ** float(g[row, j]) * float(h[dim, j]) for j in range(channels)))
            dq += math.fsum((float(da[row, j]) * float(k[j, dim]) for j in range(bt)))
            dk = math.fsum((float(v[row, j]) * 2 ** float(g[-1, j] - g[row, j]) * float(dh[dim, j]) for j in range(channels)))
            dk += math.fsum((float(da[j, row]) * float(q[j, dim]) for j in range(bt)))
            for key, expected in (('dq', dq), ('dk', dk)):
                actual = float(outputs[key][0, row, 0, dim])
                assert abs(expected - actual) < 1e-10 * (1 + abs(expected))
                errors.append(abs(expected - actual))
        for dim in (0, channels - 1):
            dv = math.fsum((float(k[row, j]) * float(dh[j, dim]) for j in range(width))) * 2 ** float(g[-1, dim] - g[row, dim])
            state_grad = math.fsum((float(h[j, dim]) * float(dh[j, dim]) for j in range(width))) * 2 ** float(g[-1, dim])
            state_grad += math.fsum((math.fsum((float(k[t, j]) * float(dh[j, dim]) for j in range(width))) * 2 ** float(g[-1, dim] - g[t, dim]) * float(v[t, dim]) for t in range(bt)))
            dgv = float(dg[row, dim]) + state_grad
            for key, expected in (('dv', dv), ('dgv', dgv)):
                actual = float(outputs[key][0, row, 0, dim])
                assert abs(expected - actual) < 1e-10 * (1 + abs(expected))
                errors.append(abs(expected - actual))
    return (outputs, dict(method='Verbatim original naive_chunk_gsa_bwd_k_dqkvg CPU64 reference; exp2 gates and all five output equations independently checked with scalar sums', reference_AST_unchanged=True, scalar_checked_values=len(errors), scalar_max_abs_difference=max(errors), original_partial_reduction_outputs=['A', 'dv', 'dgv'], scale=scale, BT=bt))

from original_cpu import naive_chunk_gsa_bwd_k_dqkvg
