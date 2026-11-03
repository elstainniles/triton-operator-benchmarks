"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    b, t, h, k = d['q'].shape
    bk, bv = (d['BK'], d['BV'])
    assert d['G'] == 1 and d['g_cumsum'] is None
    q = d['q'].double().transpose(1, 2)[..., :bk]
    key = d['k'].double().transpose(1, 2)[..., :bk]
    value = d['v'].double().transpose(1, 2)
    do = d['do'].double().transpose(1, 2)
    lse = d['lse'].double().transpose(1, 2)
    delta = d['delta'].double().transpose(1, 2)
    score = key @ q.transpose(-1, -2) * d['scale'] * 1.4426950216
    probability = torch.triu(torch.exp2(score - lse[..., None, :]))
    dv = probability @ do
    dp = value @ do.transpose(-1, -2)
    ds = probability * (dp - delta[..., None, :])
    dk = d['dk'].double().transpose(1, 2).clone()
    dk[..., :bk] = ds @ q * d['scale']
    partials = []
    for start in range(0, value.shape[-1], bv):
        dp_part = value[..., start:start + bv] @ do[..., start:start + bv].transpose(-1, -2)
        partials.append(probability * (dp_part - delta[..., None, :]) @ q * d['scale'])
    disagreement = float((partials[0] - partials[1]).abs().max())
    assert disagreement > 0 and len(partials) == 2
    for row, col in [(0, 0), (t - 1, 1)]:
        expected = math.fsum((float(probability[0, 0, row, j]) * float(do[0, 0, j, col]) for j in range(row, t)))
        errors.append(abs(float(dv[0, 0, row, col]) - expected))
    out = {'dk': dk.transpose(1, 2), 'dv': dv.transpose(1, 2)}
    method = 'CPU64 causal attention dkv using original BK32 query/key domain and full V64 derivative; unaddressed dk columns retain input prestate. Original NV2 overlapping dk writes retained and explicitly diagnosed.'
    metadata = {'source_write_race': {'grid_NV': 2, 'BK': bk, 'K': k, 'BV': bv, 'V': value.shape[-1], 'shared_written_output': 'dk[..., :32]', 'max_difference_between_i_v_partial_values': disagreement}, 'original_outputs_arguments': {'dk': 7, 'dv': 8}}
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'scalar_cross_checks': len(errors), 'scalar_max_abs_error': max(errors), **metadata})
