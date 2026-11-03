
from opbench.fla_fixtures import load_fixture
from yb_1edbd95d8265b0 import *

def test_reference(load_dir, reference_path):
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
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'o': o, 'u': u, 'final_state': final_state}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
