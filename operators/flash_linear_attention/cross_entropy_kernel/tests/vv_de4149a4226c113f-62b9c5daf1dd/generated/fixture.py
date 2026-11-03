"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.
Logical fixture names are dictionary keys; this module reads and writes no data files.
Source SHA256: 146be15ebf8d7327e6769d15f01af2353d6d4b9978737cf529d245c31bed6336
Operation: fused_linear_cross_entropy
"""
def make_fixtures(seed=20261030):
    import torch
    from pathlib import PurePosixPath
    SEED = seed
    N = 65539
    selected = op = 'fused_linear_cross_entropy'
    fixtures = {}
    folder = PurePosixPath('') / selected
    gen = torch.Generator(device='cpu').manual_seed(SEED)
    logits = torch.randn((2, 128, 1024), generator=gen)
    target = torch.randint(0, 1024, (2, 128), generator=gen, dtype=torch.int64)
    total = 256
    lse = torch.logsumexp(logits.double(), dim=-1).float()
    grad = torch.softmax(logits.double(), dim=-1)
    grad.scatter_add_(-1, target[..., None], torch.full((2, 128, 1), -1.0, dtype=torch.float64))
    grad = (grad / total).float()
    loss = ((lse.double() - logits.double().gather(-1, target[..., None]).squeeze(-1)) / total).float()
    inp = dict(logits=logits, target=target, lse=lse, total=total, ignore_index=-100, label_smoothing=0.0, logit_scale=1.0, logit_softcapping=None, reduction='mean', B=2, T=128, V=1024)
    path = folder / 'test_cross_entropy_kernel_cuda.pt'
    fixtures[path.name] = dict(input_data=inp, gpu_output=dict(loss=loss), independent_gradient_reference=grad)
    return fixtures
