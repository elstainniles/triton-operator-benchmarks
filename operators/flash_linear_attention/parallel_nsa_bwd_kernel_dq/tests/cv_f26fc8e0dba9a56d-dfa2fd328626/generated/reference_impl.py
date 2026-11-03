"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference(case, contract, data):
    cid = case['id']
    q, do = (data['q'], data['do'])
    b, t, hq, _ = q.shape
    k, v, bs = (data['k'], data['v'], data['BS_param'])
    h = k.shape[2]
    blocks = torch.zeros(b, h, t, t // bs + 1, dtype=torch.bool)
    blocks.scatter_(-1, data['block_indices'].permute(0, 2, 1, 3).clamp_max(t // bs), True)
    mask = blocks[..., :t // bs].repeat_interleave(bs, -1) & torch.ones(t, t, dtype=torch.bool).tril()[None, None]
    o, lse, delta, dq, checks = helper.attention(q, k, v, do, mask)
    assert torch.equal(lse, data['lse']) and torch.equal(delta, data['delta'])
    outputs = {'dq': dq}
    method = 'CPU64 attention gradient consuming frozen stored lse/delta, preserving explicit scaling and ds dtype conversions'
    return outputs

import math_helper as helper
