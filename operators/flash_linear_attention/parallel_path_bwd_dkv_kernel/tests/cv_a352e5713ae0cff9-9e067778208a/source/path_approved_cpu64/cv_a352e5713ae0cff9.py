
from opbench.fla_fixtures import load_fixture
from yb_a51fc6d530c908 import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_path_bwd_dkv_kernel'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    q_new_large = inputs_dict['q_new_large'].to(device)
    k_new_large = inputs_dict['k_new_large'].to(device)
    v = inputs_dict['v'].to(device)
    g_cumsum = inputs_dict['g_cumsum']
    if g_cumsum is not None:
        g_cumsum = g_cumsum.to(device)
    do = inputs_dict['do'].to(device)
    dv = inputs_dict['dv'].to(device)
    dg_cumsum = inputs_dict['dg_cumsum']
    if dg_cumsum is not None:
        dg_cumsum = dg_cumsum.to(device)
    hc_whole = inputs_dict['hc_whole'].to(device)
    scale = inputs_dict['scale']
    L = inputs_dict['L'].to(device)
    delta = inputs_dict['delta'].to(device)
    S = inputs_dict['S']
    BT = inputs_dict['BT']
    BS = inputs_dict['BS']
    dk_npu, dv_out_npu, dg_cumsum_out_npu = parallel_path_bwd_dkv_fn(q=q_new_large, k=k_new_large, v=v, g_cumsum=g_cumsum, do=do, dv=dv.clone(), dg_cumsum=dg_cumsum.clone() if dg_cumsum is not None else None, hc_whole=hc_whole, scale=scale, cu_seqlens=None, L=L, D=delta, S=S, BT=BT, BS=BS)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'dk': dk_npu, 'dv': dv_out_npu}, reference)
    assert dg_cumsum_out_npu is None, 'Original no-gate configuration must return None dg_cumsum'

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
