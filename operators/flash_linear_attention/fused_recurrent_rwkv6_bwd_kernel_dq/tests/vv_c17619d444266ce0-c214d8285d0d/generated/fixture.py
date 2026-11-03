"""Fixed-seed CPU inputs and independent references; no external tensor files."""
import torch

def rwkv6_reference(inputs, bv=64, progress=False):
    """Float64 state recursion with disjoint value-axis partial derivatives."""
    k, v, w, do, u = [inputs[name].double() for name in ('k', 'v', 'w', 'do', 'u')]
    b, t, heads, keys = k.shape
    values = v.shape[-1]
    nv = (values + bv - 1) // bv
    h0 = inputs.get('initial_state')
    state = h0.double().clone() if h0 is not None else torch.zeros(b, heads, keys, values, dtype=torch.float64)
    raw_dq = torch.empty(nv, b, t, heads, keys, dtype=torch.float32)
    raw_dq1 = torch.empty_like(raw_dq)
    dq = torch.empty(b, t, heads, keys, dtype=torch.float32)
    dq1 = torch.empty_like(dq)
    scale = float(inputs['scale'])
    order = range(t - 1, -1, -1) if inputs.get('reverse', False) else range(t)
    for step, pos in enumerate(order):
        key = k[:, pos]
        val = v[:, pos]
        grad = do[:, pos]
        outer = key[..., None] * val[..., None, :]
        total, total1 = (torch.zeros_like(key), torch.zeros_like(key))
        for part in range(nv):
            lo, hi = (part * bv, min((part + 1) * bv, values))
            h_part = state[..., lo:hi]
            g_part = grad[..., lo:hi]
            aux = (h_part * g_part[..., None, :]).sum(-1)
            partial = ((h_part + u[None, ..., None] * outer[..., lo:hi]) * g_part[..., None, :]).sum(-1) * scale
            raw_dq[part, :, pos] = partial.float()
            raw_dq1[part, :, pos] = aux.float()
            total.add_(partial)
            total1.add_(aux)
        dq[:, pos] = total.float()
        dq1[:, pos] = total1.float()
        state.mul_(w[:, pos].exp()[..., None]).add_(outer)
        if progress and (step % 256 == 0 or step == t - 1):
            print(f'rwkv6 CPU recurrence: {step + 1}/{t}', flush=True)
    return {'raw_dq': raw_dq, 'raw_dq1': raw_dq1, 'dq': dq, 'dq1': dq1, 'dq_from_stored_partials': raw_dq.sum(0), 'dq1_from_stored_partials': raw_dq1.sum(0)}

def make_fixtures():
    rng=torch.Generator(device='cpu').manual_seed(20261030)
    def randn(*shape,dtype=torch.float32):
        return torch.randn(*shape,generator=rng,dtype=dtype,device='cpu')
    shape=(4,2048,4,256)
    inputs={name:randn(*shape,dtype=torch.float16) for name in ('q','k','v','do')}
    inputs.update(w=torch.nn.functional.logsigmoid(randn(*shape)),u=randn(4,256,dtype=torch.float16),
                  initial_state=randn(4,4,256,256,dtype=torch.float16),scale=256**-0.5,reverse=False)
    payload=dict(operation='rwkv6_dq',input_origin='synthetic_cpu',seed=20261030,gpu_reference_created=False,inputs=inputs)
    return {'inputs.pt':payload,'reference.pt':rwkv6_reference(inputs)}
