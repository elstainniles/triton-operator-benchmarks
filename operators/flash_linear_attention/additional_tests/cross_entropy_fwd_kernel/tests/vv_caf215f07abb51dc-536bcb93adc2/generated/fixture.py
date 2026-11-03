"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: cross_entropy_fwd_split
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'cross_entropy_fwd_split'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    logits = torch.rand((1024, 32000), generator=gen)
    labels = torch.randint(0, 32000, (1024,), generator=gen, dtype=torch.long)
    initial = torch.zeros((2, 1024))
    lse = torch.stack((torch.logsumexp(logits[:, :16384].double(), 1), torch.logsumexp(logits[:, 16384:].double(), 1))).float()
    loss = torch.zeros_like(initial)
    rows = torch.arange(1024)
    which = (labels >= 16384).long()
    loss[which, rows] = -logits[rows, labels]
    inp = dict(loss_ptr=initial.clone(), lse_ptr=initial.clone(), z_loss_ptr=initial.clone(), logits_ptr=logits, labels_ptr=labels, label_smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, logit_softcapping=None, ignore_index=-100, total_classes=32000, class_start_idx=0, n_cols=32000, n_rows=1024, logits_row_stride=32000, BLOCK_SIZE=16384, num_warps=16, SPLIT=True)
    path = folder / 'test_cross_entropy_fwd_kernel.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(loss_ptr=loss, lse_ptr=lse, z_loss_ptr=initial.clone()), grid=(1024, 2))
    return fixtures
