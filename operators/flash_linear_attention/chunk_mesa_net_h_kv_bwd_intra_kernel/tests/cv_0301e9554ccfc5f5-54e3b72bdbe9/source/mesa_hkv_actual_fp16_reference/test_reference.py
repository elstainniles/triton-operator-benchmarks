
from opbench.fla_fixtures import load_fixture
import torch
from yb_01557b980bc783 import *

def test_cpu_reference():
    payload = load_fixture('generated-data/original_seed42_fp16_inputs.pt', map_location='cpu', weights_only=False)
    input_data = payload['input_data']
    grid = payload['grid']
    input_data_npu = convert_tensor_with_device_type(input_data, device_type='npu')
    chunk_mesa_net_h_kv_bwd_intra_kernel[grid](q_star=input_data_npu['q_star'], k=input_data_npu['k'], v=input_data_npu['v'], beta=input_data_npu['beta'], h_kv=input_data_npu['h_kv'], g=input_data_npu['g'], do=input_data_npu['do'], dh_kv=input_data_npu['dh_kv'], dq=input_data_npu['dq'], dk_beta=input_data_npu['dk'], dg=input_data_npu['dg'], dv=input_data_npu['dv'], cu_seqlens=input_data_npu['cu_seqlens'] if input_data_npu['cu_seqlens'].numel() > 0 else None, chunk_indices=input_data_npu['chunk_indices'] if input_data_npu['chunk_indices'].numel() > 0 else None, B=input_data_npu['B'], T=input_data_npu['T'], H=input_data_npu['H'], K=input_data_npu['K'], V=input_data_npu['V'], BT=input_data_npu['BT'], BK=input_data_npu['BK'], BV=input_data_npu['BV'], num_stages=1, num_warps=4)
    gold = load_fixture('generated-data/independent_outputs_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({k: input_data_npu[k] for k in ['dq', 'dk', 'dv', 'dg']}, gold)

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
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute', accuracy_criteria_changed=False, original_validation_standard='original_absolute', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
