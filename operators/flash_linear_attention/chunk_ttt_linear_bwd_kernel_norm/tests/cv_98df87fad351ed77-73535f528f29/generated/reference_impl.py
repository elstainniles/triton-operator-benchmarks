"""Existing independent CPU equations; file-generation main omitted."""
import torch
import math

def load_reference(source, name, promote=False, transpose_ttt=False):
    return (naive_chunk_ttt_linear_bwd_norm, ['Original naive _dkh @ _h changed to _dkh @ _h.T to match actual h pointer (V,K), strides (1,V)'])

def ttt(d, source):
    fn, corrections = load_reference(source, 'naive_chunk_ttt_linear_bwd_norm', transpose_ttt=True)
    q = d['q']
    b, t, h, width = q.shape
    bt = 16
    args = [d[key] for key in ('q', 'k', 'v', 'v_new', 'x', 'y', 'rstd', 'w', 'b', 'eta', 'h0', 'hb0', 'h', 'dht', 'dhbt', 'dv_new', 'do_tensor')]
    values = fn(*args, width ** (-0.5), bt, dtype=torch.float64)
    out = dict(zip(('dh', 'dhb', 'dh0', 'dhb0', 'dv', 'dk', 'dw', 'db'), values))
    start = t - bt
    k, dvn, x, y, vn = (d[key][0, start:t, 0].double() for key in ('k', 'dv_new', 'x', 'y', 'v_new'))
    dh, dhb = (d['dht'][0, 0].double(), d['dhbt'][0, 0, 0].double())
    eta = float(d['eta'][0, t - 1, 0, 0])
    w = d['w'][0].double()
    old_h = d['h'][0, -1, 0].double()
    errors = []
    scalar_dkh = []
    for row in range(bt):
        a = [float(dvn[row, j]) - eta * math.fsum((float(k[row, l]) * float(dh[l, j]) for l in range(width))) - eta * float(dhb[j]) for j in range(width)]
        r = float(d['rstd'][0, start + row, 0, 0])
        ax = math.fsum((a[j] * float(x[row, j]) for j in range(width)))
        xy = math.fsum((float(x[row, j]) * float(y[row, j]) for j in range(width)))
        dy = [r * (a[j] * width - math.fsum(a) - float(x[row, j]) * ax) / width for j in range(width)]
        dx = [-r * (a[j] * xy + float(y[row, j]) * ax) / width + float(w[j]) ** 2 * dy[j] for j in range(width)]
        dr = math.fsum((a[j] * float(vn[row, j]) / r for j in range(width)))
        xdx = math.fsum((float(x[row, j]) * dx[j] for j in range(width)))
        dkh = [r * (width * dx[j] - math.fsum(dx) - float(x[row, j]) * xdx) / width - r * r * dr * float(x[row, j]) / width for j in range(width)]
        scalar_dkh.append(dkh)
        for col in (0, width - 1):
            dv = -float(w[col]) * dy[col]
            dk = float(w[col]) * dy[col] + math.fsum((dkh[j] * float(old_h[col, j]) for j in range(width)))
            for expected, actual in ((dv, float(out['dv'][0, start + row, 0, col])), (dk, float(out['dk'][0, start + row, 0, col]))):
                error = abs(expected - actual)
                assert error <= 1e-09 * (1 + abs(expected)), (expected, actual)
                errors.append(error)
    for ki, vi in ((0, 0), (width - 1, width - 1)):
        increment = math.fsum((width ** (-0.5) * float(d['q'][0, start + i, 0, ki]) * float(d['do_tensor'][0, start + i, 0, vi]) + float(k[i, ki]) * scalar_dkh[i][vi] for i in range(bt)))
        target = float(dh[ki, vi]) + increment
        actual = float(out['dh'][0, -2, 0, ki, vi])
        assert abs(target - actual) <= 1e-09 * (1 + abs(target))
        errors.append(abs(target - actual))
    torch.testing.assert_close(out['dh'][:, -1], d['dht'].double(), rtol=0, atol=0)
    torch.testing.assert_close(out['dhb'][:, -1], d['dhbt'].double(), rtol=0, atol=0)
    return (out, dict(method='Original independent CPU64 TTT layernorm backward reference with h storage transpose corrected', reference_only_corrections=corrections, last_chunk_scalar_comparisons=len(errors), scalar_max_abs_difference=max(errors), final_state_initialization_exact=True))

from original_cpu import naive_chunk_ttt_linear_bwd_norm
