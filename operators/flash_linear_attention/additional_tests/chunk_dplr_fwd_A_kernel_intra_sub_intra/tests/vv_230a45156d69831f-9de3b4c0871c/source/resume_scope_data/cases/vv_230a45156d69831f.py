
from opbench.fla_fixtures import load_fixture
from yb_c934ac92fe2c75 import *

def test_diagnostic(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_chunk_dplr_fwd_A_kernel_intra_sub_intra_cuda.pt'), map_location='cpu', weights_only=False)
    input_data = data['input_data']
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    input_data_npu = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    Aab_npu, Aqk_npu, Aak_npu, Aqb_npu, qg_npu, kg_npu, ag_npu, bg_npu = chunk_dplr_fwd_intra(q=input_data_npu['q'], k=input_data_npu['k'], a=input_data_npu['a'], b=input_data_npu['b'], gi=input_data_npu['gi'], ge=input_data_npu['ge'], scale=scale, chunk_size=chunk_size, safe_gate=False, cu_seqlens=None, chunk_indices=None)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; independent CPU reference retained; no accuracy or timing success claimed')
