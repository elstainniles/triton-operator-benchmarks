
from opbench.fla_fixtures import load_fixture
from yb_890ee9340aad39 import *

def test_reference(load_dir, reference_path):
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs_prepare_wy.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    A_ab = inputs_dict['A_ab'].to(device)
    cu_seqlens = inputs_dict['cu_seqlens']
    chunk_size = inputs_dict['chunk_size']
    A_ab_inv = prepare_wy_repr_fwd(A_ab, cu_seqlens, chunk_size=chunk_size)
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'A_ab_inv': A_ab_inv}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
