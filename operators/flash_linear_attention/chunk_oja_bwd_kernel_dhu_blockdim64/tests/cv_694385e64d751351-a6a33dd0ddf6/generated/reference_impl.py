"""Existing independent CPU equations; file-generation main omitted."""
import reference_impl_common as common
import torch

def reference(cid, d, source=None):
    bt = int(d.get('chunk_size', 64))
    scale = float(d['scale'])
    converted = {k: v.double() if isinstance(v, torch.Tensor) and v.ndim else v.item() if isinstance(v, torch.Tensor) else v for k, v in d.items()}
    dh, dh0, dk2 = naive_chunk_oja_bwd_dhu(**converted)
    bt = 64
    q, vg, w, do, dk, gv = (common.tile(d[n], bt) for n in ('q', 'vg', 'w', 'do', 'dk', 'gv'))
    current = d['dht'].double().clone()
    parts = [None] * q.shape[1]
    keyparts = [None] * q.shape[1]
    for t in range(q.shape[1] - 1, -1, -1):
        parts[t] = current.clone()
        key = vg[:, t] @ current.transpose(-1, -2) + dk[:, t]
        keyparts[t] = key
        current = current * gv[:, t, :, -1, None, :].exp() + q[:, t].transpose(-1, -2) @ (do[:, t] * gv[:, t].exp()) * scale - key.transpose(-1, -2) @ w[:, t]
    alt = {'dh': torch.stack(parts, 1), 'dh0': current, 'dk2': common.untile(torch.stack(keyparts, 1))}
    gold = {'dh': dh, 'dh0': dh0, 'dk2': dk2}
    for key in gold:
        torch.testing.assert_close(gold[key], alt[key], rtol=1e-11, atol=1e-11)
    return gold

from original_cpu import naive_chunk_oja_bwd_dhu
