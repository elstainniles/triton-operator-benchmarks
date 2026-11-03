
from opbench.fla_fixtures import load_fixture
from yb_890ee9340aad39 import *

def test_diagnostic(load_dir):
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs_prepare_wy.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    A_ab = inputs_dict['A_ab'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    A_ab_inv = prepare_wy_repr_fwd(A_ab, cu_seqlens, chunk_size=chunk_size)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; independent CPU reference retained; no accuracy or timing success claimed')
