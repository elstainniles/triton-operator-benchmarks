
from opbench.fla_fixtures import load_fixture
from yb_0e5e2cf2748400 import *

def test_diagnostic(load_dir):
    data = load_fixture(os.path.join(load_dir, 'test_copy_last_chunk_kernel_cuda.pt'), map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    B = input_data['B']
    T = input_data['T']
    H = input_data['H']
    K = input_data['K']
    V = input_data['V']
    L = input_data['L']
    BT = input_data['BT']
    grid = (B * H,)
    copy_last_chunk_kernel[grid](q=input_data['q'], k=input_data['k'], v=input_data['v'], g=input_data['g'], level_scales=input_data['level_scales'], cu_seqlens=None, q_prev=input_data['q_prev'], k_prev=input_data['k_prev'], v_prev=input_data['v_prev'], g_prev=input_data['g_prev'], level_scales_prev=input_data['level_scales_prev'], offsets=input_data['offsets'], T=T, H=H, K=K, V=V, L=L, BT=BT)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_undefined_original_reference; independent CPU reference retained; no accuracy or timing success claimed')
