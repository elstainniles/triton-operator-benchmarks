
from opbench.fla_fixtures import load_fixture
from yb_c1149824bf6255 import *

def test_original_absolute():
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/chunk_bwd_kernel_dqke'
    q = load_fixture(os.path.join(load_dir, 'q.pt'), map_location=torch.device('cpu'))
    k = load_fixture(os.path.join(load_dir, 'k.pt'), map_location=torch.device('cpu'))
    v_new = load_fixture(os.path.join(load_dir, 'v_new.pt'), map_location=torch.device('cpu'))
    eta = load_fixture(os.path.join(load_dir, 'eta.pt'), map_location=torch.device('cpu'))
    h = load_fixture(os.path.join(load_dir, 'h.pt'), map_location=torch.device('cpu'))
    do_tensor = load_fixture(os.path.join(load_dir, 'do_tensor.pt'), map_location=torch.device('cpu'))
    dh = load_fixture(os.path.join(load_dir, 'dh.pt'), map_location=torch.device('cpu'))
    dhb = load_fixture(os.path.join(load_dir, 'dhb.pt'), map_location=torch.device('cpu'))
    sc = load_fixture(os.path.join(load_dir, 'scale.pt')).item()
    cs = load_fixture(os.path.join(load_dir, 'chunk_size.pt')).item()
    from yb_7bff0dd87eff27 import chunk_ttt_linear_bwd_dqke
    q_npu = q.to(DEVICE)
    k_npu = k.to(DEVICE)
    v_new_npu = v_new.to(DEVICE)
    eta_npu = eta.to(DEVICE)
    h_npu = h.to(DEVICE)
    do_npu = do_tensor.to(DEVICE)
    dh_npu = dh.to(DEVICE)
    dhb_npu = dhb.to(DEVICE)
    npu_dq, npu_dk, npu_de = chunk_ttt_linear_bwd_dqke(q_npu, k_npu, v_new_npu, eta_npu, h_npu, do_npu, dh_npu, dhb_npu, sc, None, cs)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_original_cpu_reference({'dq': npu_dq, 'dk': npu_dk, 'de': npu_de}, reference)

def compare_original_cpu_reference(outputs, references):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        actual = output.to('cpu').double().float()
        expected = references[name].float()
        assert references[name].dtype == torch.float64
        diff = (actual.double() - expected.double()).abs()
        allowed = 0.05 + 0.05 * expected.double().abs()
        record = dict(output=name, dtype=str(output.dtype), rtol=0.05, atol=0.05, reference_dtype='float64', comparison_dtype='float32', elements=actual.numel(), max_abs_error=float(diff.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (diff > allowed)).sum()))
        try:
            assert torch.allclose(actual, expected, atol=0.05, rtol=0.05), 'original CPU64 absolute comparison failed: ' + name
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('original_cpu_accuracy.json').write_text(json.dumps(dict(validation_standard='original_cpu_absolute', accuracy_criteria_changed=False, outputs=records), indent=2) + '\n')
    assert not failures, 'Original CPU reference precision failed: ' + ', '.join(failures)
