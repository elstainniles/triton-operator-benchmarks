
from opbench.fla_fixtures import load_fixture
from yb_1cc9236d76265c import *

def test_cpu_fp64_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_simple_gla_bwd_kernel/test_parallel_simple_gla_bwd_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    device = torch.device('npu')
    input_data = convert_tensor_with_device_type(data['input_data'], device_type=device)
    q = input_data['q']
    k = input_data['k']
    v = input_data['v']
    g = input_data['g']
    do = input_data['do']
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    cu_seqlens = input_data['cu_seqlens']
    chunk_indices = input_data['chunk_indices']
    dq, dk, dv, dg = parallel_simple_gla_bwd(q=q, k=k, v=v, g=g, do=do, scale=scale, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dq': dq, 'dk': dk, 'dv': dv, 'dg': dg}, reference)

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
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute_cpu64_reference', accuracy_criteria_changed=False, original_validation_standard='original_dtype_absolute', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
