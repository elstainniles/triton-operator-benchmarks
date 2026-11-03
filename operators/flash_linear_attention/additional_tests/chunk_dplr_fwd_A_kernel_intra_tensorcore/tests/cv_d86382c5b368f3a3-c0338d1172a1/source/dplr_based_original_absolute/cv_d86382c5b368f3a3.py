
from opbench.fla_fixtures import load_fixture
from yb_c2e550c04501c8 import *

def test_cpu_fp64_absolute():
    ptfile_path = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_dplr_fwd_A_kernel_intra_tensorcore/test_chunk_dplr_fwd_A_kernel_intra_tensorcore_cuda.pt'
    data = load_fixture(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    input_data = convert_tensor_with_device_type(data['input_data'], device_type='npu')
    scale = input_data['scale']
    chunk_size = input_data['chunk_size']
    Aab, Aqk, Aak, Aqb, qg, kg, ag, bg = chunk_dplr_fwd_intra(q=input_data['q'], k=input_data['k'], a=input_data['a'], b=input_data['b'], gi=input_data['gi'], ge=input_data['ge'], scale=scale, chunk_size=chunk_size, safe_gate=True, cu_seqlens=None, chunk_indices=None)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'Aab': Aab, 'Aqk': Aqk, 'Aak': Aak, 'Aqb': Aqb, 'qg': qg, 'kg': kg, 'ag': ag, 'bg': bg}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = {'float32': 1e-05, 'float16': 0.001, 'bfloat16': 0.01}[dtype]
        actual = output.detach().cpu()
        expected = references[name].cpu().to(output.dtype)
        assert references[name].dtype == torch.float64
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='float64', reference_rounded_to_output_dtype=True, comparison_dtype=dtype)
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            validate_cmp(dtype=dtype, y_cal=actual, y_ref=expected, device_type='cpu')
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('original_absolute_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute_cpu_reference_v1', accuracy_criteria_changed=False, original_validation_standard='original_absolute_dtype_tolerance', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
