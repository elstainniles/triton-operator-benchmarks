
from opbench.fla_fixtures import load_fixture
from yb_1e92f1276a5cee import *

def test_cpu_fp64_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_gsa_bwd_k_kernel_dqkvg'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    k = load_fixture(os.path.join(load_dir, 'k.pt'), map_location=torch.device('cpu'))
    v = load_fixture(os.path.join(load_dir, 'v.pt'), map_location=torch.device('cpu'))
    g = load_fixture(os.path.join(load_dir, 'g.pt'), map_location=torch.device('cpu'))
    h = load_fixture(os.path.join(load_dir, 'h.pt'), map_location=torch.device('cpu'))
    do = load_fixture(os.path.join(load_dir, 'do.pt'), map_location=torch.device('cpu'))
    dh = load_fixture(os.path.join(load_dir, 'dh.pt'), map_location=torch.device('cpu'))
    dA = load_fixture(os.path.join(load_dir, 'dA.pt'), map_location=torch.device('cpu'))
    dg = load_fixture(os.path.join(load_dir, 'dg.pt'), map_location=torch.device('cpu'))
    scale = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    BT = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    from yb_b16ff00ec8999a import chunk_gsa_bwd_k_kernel_dqkvg
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_npu = v.to(DEVICE)
    g_npu = g.to(DEVICE)
    h_npu = h.to(DEVICE)
    do_npu = do.to(DEVICE)
    dh_npu = dh.to(DEVICE)
    dA_npu = dA.to(DEVICE)
    dg_npu = dg.to(DEVICE)
    B_val, T_val, HQ_val, K_val = q_npu.shape
    H_val = k_npu.shape[2]
    V_val = v_npu.shape[-1]
    BK = min(64, triton.next_power_of_2(K_val))
    BV = min(64, triton.next_power_of_2(V_val))
    NK = triton.cdiv(K_val, BK)
    NG = HQ_val // H_val
    NT = triton.cdiv(T_val, BT)
    A_npu = torch.empty(NK, B_val, T_val, HQ_val, BT, dtype=q_npu.dtype, device=DEVICE)
    dq_npu = torch.empty(B_val, T_val, HQ_val, K_val, dtype=q_npu.dtype, device=DEVICE)
    dk_npu = torch.empty(B_val, T_val, HQ_val, K_val, dtype=q_npu.dtype, device=DEVICE)
    dv_npu = torch.empty(NK, B_val, T_val, HQ_val, V_val, dtype=q_npu.dtype, device=DEVICE)
    dgv_npu = torch.empty(NK, B_val, T_val, HQ_val, V_val, dtype=torch.float32, device=DEVICE)
    grid = (NK, NT, B_val * HQ_val)
    chunk_gsa_bwd_k_kernel_dqkvg[grid](q_npu, k_npu, v_npu, h_npu, g_npu, A_npu, do_npu, dh_npu, dq_npu, dk_npu, dv_npu, dg_npu, dgv_npu, dA_npu, cu_seqlens=None, chunk_indices=None, scale=scale, T=T_val, B=B_val, HQ=HQ_val, H=H_val, K=K_val, V=V_val, BT=BT, BK=BK, BV=BV, NG=NG)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'A': A_npu.sum(0), 'dq': dq_npu, 'dk': dk_npu, 'dv': dv_npu.sum(0), 'dgv': dgv_npu.sum(0)}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = 0.01
        actual = output.detach().cpu()
        expected = references[name].cpu().float()
        assert references[name].dtype == torch.float64
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='float64', reference_rounded_to_output_dtype=True, comparison_dtype='float32')
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            assert torch.allclose(actual, expected, rtol=tolerance, atol=tolerance), 'Original torch.allclose failed: ' + name
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False)
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('original_absolute_accuracy.json').write_text(json.dumps(dict(validation_standard='original_absolute_cpu_reference_v1', accuracy_criteria_changed=False, original_validation_standard='original_absolute_dtype_tolerance', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 absolute validation failed for outputs: ' + ', '.join(failures)
