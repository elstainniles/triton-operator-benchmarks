"""Existing independent CPU equations; file-generation main omitted."""
import math
import torch

def reference(cid, d):
    errors = []
    q, w1, w2 = (d[n].double() for n in ['q', 'w1', 'w2'])
    b, t, hq, k = q.shape
    h = w1.shape[2]
    s, bt, bs = (d['S'], d['BT'], d['BS'])
    assert d['cu_seqlens'] is None and t % bt == 0
    result = torch.zeros(b, t, math.ceil(t / s), hq, k, dtype=torch.float64)
    w1 = w1.repeat_interleave(hq // h, dim=2).transpose(1, 2)
    w2 = w2.repeat_interleave(hq // h, dim=2).transpose(1, 2)
    for start in range(0, t, bt):
        q0 = q[:, start:start + bt].transpose(1, 2)
        current = q0.clone()
        if bs == bt and start % s == 0:
            result[:, start:start + bt, start // s] = current.transpose(1, 2)
        for offset in range(start + bt - 2 * bs, s - bs, -bs):
            residual = torch.eye(k, dtype=torch.float64) - w1[:, :, offset:offset + bs].transpose(-1, -2) @ w2[:, :, offset:offset + bs]
            update = current @ residual
            written = torch.arange(start, start + bt) >= offset + bs
            current = torch.where(written[None, None, :, None], update, current)
            if offset % s == 0:
                result[:, start:start + bt, offset // s] = current.transpose(1, 2)
        direct = q0.clone()
        for offset in range(start + bt - 2 * bs, s - bs, -bs):
            cross = direct @ w1[:, :, offset:offset + bs].transpose(-1, -2)
            cross = torch.where((torch.arange(start, start + bt) >= offset + bs)[None, None, :, None], cross, 0)
            direct -= cross @ w2[:, :, offset:offset + bs]
        errors.append(float((current - direct).abs().max()) / max(1, float(current.abs().max())))
    out = {'q_new': result}
    method = 'CPU64 product of independently formed residual matrices I-w1T@w2, all written and zero-initialized output blocks; cross-checked rank updates'
    assert all((x.dtype == torch.float64 and torch.isfinite(x).all() for x in out.values()))
    assert max(errors) < 1e-08, errors
    return (out, {'method': method, 'independent_cross_checks': len(errors), 'maximum_cross_check_error': max(errors)})
