
from opbench.fla_fixtures import load_fixture
from yb_0e5e2cf2748400 import *

def test_defined_outputs(load_dir):
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
    defined_rows = data['defined_rows']
    actual = {name: input_data[name][:, :defined_rows].detach().cpu() for name in data['cpu_reference']}
    compare_data_precision(data['cpu_reference'], actual, device_type='cpu')
    for name in actual:
        assert torch.equal(actual[name], data['cpu_reference'][name]), 'copy kernel changed defined input values: ' + name
    assert torch.equal(input_data['offsets'].cpu(), data['input_data']['offsets']), 'copy kernel unexpectedly changed offsets'
