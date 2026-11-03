
from opbench.fla_fixtures import load_fixture
from yb_fbb62dba127526 import *

def test_diagnostic():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_based_bwd/test_parallel_based_bwd_kernel_v2.pt'
    data = load_fixture(ptfile_path, map_location='cpu', weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    parallel_based_bwd_kernel[data['grid']](input_data['q'], input_data['k'], input_data['v'], input_data['do'], input_data['dz'], input_data['dq'], input_data['dk'], input_data['dv'], scale=input_data['scale'], T=input_data['T'], B=input_data['B'], H=input_data['H'], K=input_data['K'], V=input_data['V'], BTL=input_data['BTL'], BTS=input_data['BTS'], BK=input_data['BK'], BV=input_data['BV'])
    reference = load_fixture('generated-data/reference_original_fp32_acceptance.pt', map_location='cpu', weights_only=False)
    for key in ('dq', 'dk', 'dv'):
        npu_tensor = input_data[key].float()
        golden_tensor = reference[key].to(npu_tensor.device).float()
        assert torch.allclose(npu_tensor, golden_tensor, 0.005, 0.005)
        torch.testing.assert_close(npu_tensor, golden_tensor, rtol=0.005, atol=0.005)
