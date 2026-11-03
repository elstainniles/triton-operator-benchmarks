"""Existing independent CPU equations; file-generation main omitted."""
import torch.nn.functional as F
import torch

def make_fixtures():
    B, T, H, BT, cid = (2, 500, 4, 32, 'cv_27eb4c65ae93688c')
    gen = torch.Generator(device='cpu').manual_seed(20261030 + BT)
    keys = F.normalize(torch.randn((B, H, T, 64), generator=gen), dim=-1)
    padded = F.pad(keys, (0, 0, 0, -T % BT)).reshape(B, H, -1, BT, 64)
    a = (padded @ padded.transpose(-1, -2)).tril(-1)
    eye = torch.eye(BT, dtype=torch.float64)
    inverse = torch.linalg.solve_triangular(eye + a.double(), eye.expand(*a.shape), upper=False)
    inputs = a.reshape(B, H, -1, BT)[:, :, :T, :].transpose(1, 2).contiguous()
    output = inverse.float().reshape(B, H, -1, BT)[:, :, :T, :].transpose(1, 2).contiguous()
    return {'input_A_input.pt': inputs, 'output.pt': output}
