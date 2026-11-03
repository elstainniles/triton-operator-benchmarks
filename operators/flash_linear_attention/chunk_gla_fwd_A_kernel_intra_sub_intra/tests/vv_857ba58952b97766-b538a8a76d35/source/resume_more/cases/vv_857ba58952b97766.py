
from opbench.fla_fixtures import load_fixture
from yb_e5e4bcc8ac44fb import *

def test_diagnostic(load_dir):
    B, T, H, K = (2, 2048, 8, 128)
    chunk_size = 64
    scale = K ** (-0.5)
    device = torch.device('npu:0')
    q = load_fixture(os.path.join(load_dir, 'q.pt')).to(device)
    k = load_fixture(os.path.join(load_dir, 'k.pt')).to(device)
    g = load_fixture(os.path.join(load_dir, 'g.pt')).to(device)
    A = chunk_gla_fwd_intra_gk(q, k, g, scale, chunk_size=chunk_size)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_partial_output_binding; independent CPU reference retained; no accuracy or timing success claimed')
