
from opbench.fla_fixtures import load_fixture
from yb_1edbd95d8265b0 import *

def test_diagnostic(load_dir):
    device = torch.device('npu:0')
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    beta = inputs_dict['beta'].to(device)
    scale = inputs_dict['scale']
    initial_state = inputs_dict['initial_state'].to(device)
    output_final_state = inputs_dict['output_final_state']
    o, u, final_state = fused_recurrent_delta_rule_fwd(q=q, k=k, v=v, beta=beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state)
    raise RuntimeError('DIAGNOSTIC_LAUNCH_FINISHED: blocked_missing_gpu_comparison_reference; independent CPU reference retained; no accuracy or timing success claimed')
