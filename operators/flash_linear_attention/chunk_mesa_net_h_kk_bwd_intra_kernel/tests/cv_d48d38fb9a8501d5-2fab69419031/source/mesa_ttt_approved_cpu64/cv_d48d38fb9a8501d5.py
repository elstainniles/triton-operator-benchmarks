
from opbench.fla_fixtures import load_fixture
from yb_143ec199a07895 import *

def test_cpu_fp64_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_net_h_kk_bwd_intra_kernel/test_chunk_mesa_net_h_kk_bwd_intra_kernel_fp16_v2.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data_cpu = data['input_data']
    input_data_npu = convert_tensor_with_device_type(input_data_cpu, device_type='npu')
    chunk_mesa_net_h_kk_bwd_intra_kernel[data['grid']](k=input_data_npu['k'], h=input_data_npu['h'], dh=input_data_npu['dh'], g=input_data_npu['g'], q_star=input_data_npu['q_star'], beta=input_data_npu['beta'], dbeta=input_data_npu['dbeta'], dq=input_data_npu['dq'], dk=input_data_npu['dk'], dk_beta=input_data_npu['dk_beta'], dg=input_data_npu['dg'], dlamb=input_data_npu['dlamb'], cu_seqlens=None, chunk_indices=None, B=input_data_npu['B'], T=input_data_npu['T'], H=input_data_npu['H'], K=input_data_npu['K'], V=input_data_npu['V'], BT=input_data_npu['BT'], BK=input_data_npu['BK'], BV=input_data_npu['BV'])
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dk': input_data_npu['dk'], 'dg': input_data_npu['dg'], 'dbeta': input_data_npu['dbeta'], 'dlamb': input_data_npu['dlamb']}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = {'float32': 1e-05, 'float16': 0.001, 'bfloat16': 0.01}[dtype]
        actual = output.detach().cpu().double()
        expected = references[name].cpu()
        assert expected.dtype == torch.float64
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='float64', reference_rounded_to_output_dtype=False)
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False)
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='cpu_fp64_absolute_v1', accuracy_criteria_changed=True, original_validation_standard='cuda_error_ratio', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
