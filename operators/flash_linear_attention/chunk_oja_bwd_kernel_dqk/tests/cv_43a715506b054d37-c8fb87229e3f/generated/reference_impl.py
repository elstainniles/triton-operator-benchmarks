"""Existing independent CPU equations; file-generation main omitted."""
import math
import reference_impl_common as common
import torch

def reference(cid, d, source=None):
    bt = int(d.get('chunk_size', 64))
    scale = float(d['scale'])
    q, k, da, gv, do = (common.tile(d[n], bt) for n in ('q', 'k_new', 'dA', 'gv_cumsum', 'do'))
    state = d['h'].double()
    qscaled = (q * scale).to(d['q'].dtype).double()
    dogated = (do * gv.exp() * scale).to(d['do'].dtype).double()
    dad = da.to(d['q'].dtype).double()
    A = (qscaled @ k.transpose(-1, -2)).tril()
    dq = dogated @ state.to(d['do'].dtype).double().transpose(-1, -2) + dad @ k
    dk = dad.transpose(-1, -2) @ q
    qv = q[0, 0, 0].clone().requires_grad_(True)
    kv = k[0, 0, 0].clone().requires_grad_(True)
    cross = dogated[0, 0, 0] @ state[0, 0, 0].to(d['do'].dtype).double().T
    loss = (qv @ kv.T * dad[0, 0, 0]).sum() + (qv * cross).sum()
    aq, ak = torch.autograd.grad(loss, (qv, kv))
    torch.testing.assert_close(aq, dq[0, 0, 0], rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(ak, dk[0, 0, 0], rtol=1e-11, atol=1e-11)
    errors = []
    for i, j in [(0, 0), (bt - 1, 0), (bt - 1, bt - 1)]:
        scalar = math.fsum((float(qscaled[0, 0, 0, i, s]) * float(k[0, 0, 0, j, s]) for s in range(k.shape[-1])))
        errors.append(abs(float(A[0, 0, 0, i, j]) - scalar))
    assert max(errors) < 1e-09
    return {'A': common.untile(A), 'dq': common.untile(dq), 'dk': common.untile(dk)}
