
from opbench.fla_fixtures import load_fixture
from yb_ef38887c13c913 import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/test_chunk_kda_bwd_kernel_wy_dqkg_fused'
    inputs_dict = load_fixture(os.path.join(load_dir, 'inputs.pt'))
    device = torch.device('npu')
    q = inputs_dict['q'].to(device)
    k = inputs_dict['k'].to(device)
    v = inputs_dict['v'].to(device)
    v_new = inputs_dict['v_new'].to(device)
    g = inputs_dict['g'].to(device)
    beta = inputs_dict['beta'].to(device)
    A = inputs_dict['A'].to(device)
    h = inputs_dict['h'].to(device)
    do = inputs_dict['do'].to(device)
    dh = inputs_dict['dh'].to(device)
    dv = inputs_dict['dv'].to(device)
    state_v_first = inputs_dict['state_v_first']
    scale = inputs_dict['scale']
    chunk_size = inputs_dict['chunk_size']
    B, T, H, K = k.shape
    HV, V = (v.shape[2], v.shape[-1])
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    dq_npu = g.new_empty(B, T, HV, K, dtype=torch.float)
    dk_npu = g.new_empty(B, T, HV, K, dtype=torch.float)
    dv2_npu = torch.empty_like(v)
    dg_npu = torch.empty_like(g, dtype=torch.float)
    db_npu = torch.empty_like(beta, dtype=torch.float)
    dA_npu = torch.empty_like(A, dtype=torch.float)
    grid = (NT, B * HV)
    chunk_kda_bwd_kernel_wy_dqkg_fused[grid](q=q, k=k, v=v, v_new=v_new, g=g, beta=beta, A=A, h=h, do=do, dh=dh, dq=dq_npu, dk=dk_npu, dv=dv, dv2=dv2_npu, dg=dg_npu, db=db_npu, dA=dA_npu, cu_seqlens=None, chunk_indices=None, scale=scale, T=T, H=H, HV=HV, K=K, V=V, BT=BT, STATE_V_FIRST=state_v_first)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_original_absolute({'dq': dq_npu, 'dk': dk_npu, 'dv2': dv2_npu, 'dg': dg_npu, 'db': db_npu, 'dA': dA_npu}, reference, 0.001)

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

def compare_original_absolute(outputs, references, tolerance):
    failures = []
    for name, output in outputs.items():
        try:
            torch.testing.assert_close(output.detach().cpu().double(), references[name].cpu(), rtol=tolerance, atol=tolerance, equal_nan=True)
        except AssertionError as error:
            failures.append(name + ': ' + str(error))
    if failures:
        raise AssertionError('\n'.join(failures))
