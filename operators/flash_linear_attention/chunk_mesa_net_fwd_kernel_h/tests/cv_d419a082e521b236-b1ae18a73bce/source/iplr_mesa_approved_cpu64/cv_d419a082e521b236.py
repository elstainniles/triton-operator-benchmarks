
from opbench.fla_fixtures import load_fixture
from yb_651b9afa419e38 import *
from yb_651b9afa419e38 import _run_npu_test

def test_cpu_fp64_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_net_fwd_kernel_h/test_chunk_mesa_net_fwd_kernel_h_fp16_v2.pt'
    input_data = _run_npu_test(ptfile_path, 'fp16')
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({name: input_data[name] for name in ('h', 'h_kv', 'h_final', 'h_kv_final')}, reference)

def _run_npu_test(ptfile_path: str, dtype_label: str):
    try:
        data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file {ptfile_path} failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    chunk_mesa_net_fwd_kernel_h[data['grid']](k=input_data['k'], v=input_data['v'], beta=input_data['beta'], g=input_data['g'], h=input_data['h'], h_kv=input_data['h_kv'], h_init=input_data['h_init'], h_kv_init=input_data['h_kv_init'], h_final=input_data['h_final'], h_kv_final=input_data['h_kv_final'], cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None, split_offsets=input_data['split_offsets'] if input_data['split_offsets'].numel() > 0 else None, T=input_data['T'], H=input_data['H'], K=input_data['K'], V=input_data['V'], BT=input_data['BT'], BS=input_data['BS'], BK=input_data['BK'], BV=input_data['BV'])
    return input_data

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
