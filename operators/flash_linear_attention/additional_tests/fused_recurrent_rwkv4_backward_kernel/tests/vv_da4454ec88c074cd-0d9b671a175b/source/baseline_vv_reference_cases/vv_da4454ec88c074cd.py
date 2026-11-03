
from opbench.fla_fixtures import load_fixture
from yb_421a9babfbf0e3 import *

def test_reference(load_dir, reference_path):
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
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'gw': gw, 'gu': gu, 'gk': gk, 'gv': gv, 'gstate': gstate}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
