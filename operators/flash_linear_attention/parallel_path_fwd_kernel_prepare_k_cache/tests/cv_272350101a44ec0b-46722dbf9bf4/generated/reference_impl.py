"""Existing independent CPU equations; file-generation main omitted."""
import torch

def cache_reference(d, source):
    assert d['cu_seqlens'] is None and d['use_cache'] is True
    result = prepare_k_cache_torch_reference(d['k'].double(), d['w1'].double(), d['w2'].double(), d['BS'])
    k, w1, w2 = (d[key].double() for key in ('k', 'w1', 'w2'))
    b, t, h, width = k.shape
    bs = d['BS']
    differences = []
    for bi in range(b):
        for hi in range(h):
            for start in (0, t - bs):
                transform = torch.eye(width, dtype=torch.float64)
                for later in range(start + bs, t, bs):
                    transform = transform @ (torch.eye(width, dtype=torch.float64) - w2[bi, later:later + bs, hi].T @ w1[bi, later:later + bs, hi])
                check = k[bi, start:start + bs, hi] @ transform
                torch.testing.assert_close(result[bi, start:start + bs, hi], check, rtol=1e-10, atol=1e-10)
                differences.append(float((result[bi, start:start + bs, hi] - check).abs().max()))
    return ({'k_new': result}, {'method': 'Original pure Torch k-cache recurrence promoted to FP64; checked against product of I-w2.T@w1', 'matrix_product_checks': differences})

from original_cpu import prepare_k_cache_torch_reference
