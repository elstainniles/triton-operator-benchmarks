
from opbench.fla_fixtures import load_fixture
from yb_421a9babfbf0e3 import *

def test_diagnostic(load_dir):
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs_rwkv4_backward.pt'), map_location='cpu', weights_only=False)
    device = torch.device('npu')
    w_processed = inputs_dict['w_processed'].to(device)
    u = inputs_dict['u'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    state = inputs_dict['state'].to(device)
    grad_wkv = inputs_dict['grad_wkv'].to(device)
    grad_state = inputs_dict['grad_state'].to(device)
    gw, gu, gk, gv, gstate = fused_recurrent_rwkv4_backward(w_processed, u, k, v, state, grad_wkv, grad_state)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; independent CPU reference retained; no accuracy or timing success claimed')
