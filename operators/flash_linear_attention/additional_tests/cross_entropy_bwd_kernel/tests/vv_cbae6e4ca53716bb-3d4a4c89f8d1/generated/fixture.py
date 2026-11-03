"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: cross_entropy_bwd
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'cross_entropy_bwd'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    dloss = torch.rand((1024,), generator=gen)
    logits = torch.rand((1024, 32000), generator=gen)
    labels = torch.randint(0, 32000, (1024,), generator=gen, dtype=torch.long)
    lse = torch.logsumexp(logits, dim=-1)
    xx = logits.double().requires_grad_(True)
    losses = torch.nn.functional.cross_entropy(xx, labels, reduction='none')
    dlogits, = torch.autograd.grad(losses, xx, dloss.double())
    inp = dict(dlogits_ptr=torch.zeros_like(logits), dloss_ptr=dloss, logits_ptr=logits, lse_ptr=lse, labels_ptr=labels, label_smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, logit_softcapping=None, ignore_index=-100, total_classes=32000, class_start_idx=0, n_cols=32000, logits_row_stride=32000, dlogits_row_stride=32000, dloss_row_stride=1, BLOCK_SIZE=4096, num_warps=8)
    path = folder / 'test_cross_entropy_bwd_kernel.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(dlogits_ptr=dlogits.float()), grid=(1024, 8))
    return fixtures
