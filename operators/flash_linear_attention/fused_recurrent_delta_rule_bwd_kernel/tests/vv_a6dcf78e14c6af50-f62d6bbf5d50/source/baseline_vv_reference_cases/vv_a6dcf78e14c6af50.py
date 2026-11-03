
from opbench.fla_fixtures import load_fixture
from yb_3bbb884047bf2c import *

def test_reference(load_dir, reference_path):
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    scale = inputs_dict['scale']
    device_npu = torch.device('npu:0')
    q_npu = inputs_dict['q'].to(device_npu)
    k_npu = inputs_dict['k'].to(device_npu)
    v_npu = inputs_dict['v'].to(device_npu)
    beta_npu = inputs_dict['beta'].to(device_npu)
    do_npu = inputs_dict['do'].to(device_npu)
    dht_npu = inputs_dict['dht'].to(device_npu)
    initial_state_npu = inputs_dict['initial_state'].to(device_npu)
    initial_state_npu.requires_grad = True
    dq_npu, dk_npu, dv_npu, db_npu, dh0_npu = fused_recurrent_delta_rule_bwd(q=q_npu, k=k_npu, v=v_npu, beta=beta_npu, dht=dht_npu, do=do_npu, scale=scale, initial_state=initial_state_npu)
    reference = load_fixture(reference_path, map_location='cpu', weights_only=False)
    values = {'dq': dq_npu, 'dk': dk_npu, 'dv': dv_npu, 'db': db_npu, 'dh0': dh0_npu}
    from baseline_reference_checks import compare_cpu64
    compare_cpu64(values, reference)
