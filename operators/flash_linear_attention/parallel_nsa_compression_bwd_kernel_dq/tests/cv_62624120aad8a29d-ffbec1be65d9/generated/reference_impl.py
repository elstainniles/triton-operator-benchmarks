"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(case, contract, data):
    cid = case['id']
    q, do = (data['q'], data['do'])
    b, t, hq, _ = q.shape
    k, v, bs = (data['k_cmp'], data['v_cmp'], data['block_size'])
    mask = torch.arange(t // bs)[None] < (torch.arange(t)[:, None] + 1) // bs
    o, lse, delta, dq, checks = helper.attention(q, k, v, do, mask)
    assert torch.equal(o, data['o_cmp']) and torch.equal(lse, data['lse_cmp']) and torch.equal(delta, data['delta_cmp'])
    outputs = {'dq': dq}
    method = 'CPU64 attention gradient consuming frozen stored lse/delta, preserving explicit scaling and ds dtype conversions'
    return outputs

import math_helper as helper
