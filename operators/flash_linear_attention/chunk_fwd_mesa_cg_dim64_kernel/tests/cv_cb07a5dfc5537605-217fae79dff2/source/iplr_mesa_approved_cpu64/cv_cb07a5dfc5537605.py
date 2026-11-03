
from opbench.fla_fixtures import load_fixture
from yb_4f08bd3c58af0a import *

def test_cpu_fp64_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_cg_fwd/test_chunk_mesa_cg_dim64_kernel_v2.pt'
    ptfile_path1 = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_mesa_cg_fwd/test_chunk_mesa_cg_dim64_kernel_fp64_v2.pt'
    try:
        data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
        datafp64 = load_fixture(ptfile_path1, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f'load file {ptfile_path} failed: {str(e)}')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type=device)
    chunk_fwd_mesa_cg_dim64_kernel[data['grid']](q=input_data['q'], q_final=input_data['q_final'], k=input_data['k'], h=input_data['h'], o=input_data['o'], v=input_data['v'], h_kv=input_data['h_kv'], g=input_data['g'], beta=input_data['beta'], lamb=input_data['lamb'], cu_seqlens=input_data['cu_seqlens'] if input_data['cu_seqlens'].numel() > 0 else None, chunk_indices=input_data['chunk_indices'] if input_data['chunk_indices'].numel() > 0 else None, T=input_data['T'], max_CG_iteration=input_data['max_CG_iteration'], H=input_data['H'], K=input_data['K'], BT=input_data['BT'], BK=input_data['BK'])
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({name: input_data[name] for name in ('q_final', 'o')}, reference)

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
