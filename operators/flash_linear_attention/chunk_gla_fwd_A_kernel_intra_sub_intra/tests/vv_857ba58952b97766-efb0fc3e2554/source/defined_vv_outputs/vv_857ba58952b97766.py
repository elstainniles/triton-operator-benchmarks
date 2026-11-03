
from opbench.fla_fixtures import load_fixture
from yb_e5e4bcc8ac44fb import *

def test_defined_outputs(load_dir, reference_path):
    B, T, H, K = (2, 2048, 8, 128)
    chunk_size = 64
    scale = K ** (-0.5)
    device = torch.device('npu:0')
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(device)
    k = load_fixture(os.path.join(load_dir, 'k.pt')).to(device)
    g = load_fixture(os.path.join(load_dir, 'g.pt')).to(device)
    BT = chunk_size
    BC = min(16, BT)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.full((B, T, H, BT), 0.375, dtype=torch.float32, device=device)
    before = A.detach().cpu().clone()
    grid = (triton.cdiv(T, BT), triton.cdiv(BT, BC), B * H)
    chunk_gla_fwd_A_kernel_intra_sub_intra[grid](q=q, k=k, g=g, A=A, cu_seqlens=None, chunk_indices=None, scale=scale, T=T, H=H, K=K, BT=BT, BC=BC, BK=BK)
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    actual = A.detach().cpu()
    mask = reference['written_mask']
    validate_cmp('float16', actual[mask], reference['expected'][mask].to(actual.dtype), device_type='cpu')
    assert torch.equal(actual[~mask], before[~mask]), 'Original target changed an address outside every tl.store mask'
