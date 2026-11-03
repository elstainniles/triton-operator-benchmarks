
from opbench.fla_fixtures import load_fixture
from yb_78630b6f827ef9 import *

def test_cpu_fp64_absolute():
    device = torch.device('npu:0')
    load_dir = '/data/PytorchFile/Inference_or_Training_solution/fla_cv_ops/parallel_nsa_kernel_topk_fp16'
    data_dict = load_fixture(f'{load_dir}/parallel_nsa_kernel_topk.pt')
    input_dict = data_dict['input_data']
    q = input_dict['q'].to(device)
    k_cmp = input_dict['k_cmp'].to(device)
    lse_cmp = input_dict['lse'].to(device)
    scale = input_dict['scale']
    cu_seqlens = input_dict['cu_seqlens'].to(device) if input_dict['cu_seqlens'] is not None else None
    token_indices = input_dict['token_indices'].to(device) if input_dict['token_indices'] is not None else None
    chunk_offsets = input_dict['chunk_offsets'].to(device) if input_dict['chunk_offsets'] is not None else None
    B = input_dict['B']
    T = input_dict['T']
    H = input_dict['H']
    HQ = input_dict['HQ']
    G = input_dict['G']
    K = input_dict['K']
    S = input_dict['S']
    BC = input_dict['BC']
    BS = input_dict['BS']
    BK = input_dict['BK']
    grid = (T, B * H)
    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)
    parallel_nsa_kernel_topk[grid](q=q.clone(), k=k_cmp.clone(), lse=lse_cmp.clone(), scale=scale, block_indices=block_indices, cu_seqlens=cu_seqlens, token_indices=token_indices, chunk_offsets=chunk_offsets, T=T, H=H, HQ=HQ, G=G, K=K, S=S, BC=BC, BS=BS, BK=BK)
    reference = load_fixture('generated-data/reference_fp64.pt', map_location='cpu', weights_only=False)
    compare_cpu64_outputs({'block_indices': block_indices}, reference)

def compare_cpu64_outputs(outputs, references, masks=None):
    import json
    from pathlib import Path
    records = []
    failures = []
    for name, output in outputs.items():
        dtype = str(output.dtype).removeprefix('torch.')
        tolerance = 0.0
        actual = output.detach().cpu()
        expected = references[name].cpu()
        assert expected.dtype == torch.int32 and actual.dtype == torch.int32
        defined = torch.ones_like(expected, dtype=torch.bool) if masks is None else masks[name]
        actual, expected = (actual[defined], expected[defined])
        difference = (actual - expected).abs()
        allowed = tolerance + tolerance * expected.abs()
        record = dict(output=name, dtype=dtype, rtol=tolerance, atol=tolerance, defined_elements=actual.numel(), undefined_elements=output.numel() - actual.numel(), max_abs_error=float(difference.max()), mismatched_elements=int((~torch.isfinite(actual) | ~torch.isfinite(expected) | (difference > allowed)).sum()), reference_dtype='int32', reference_rounded_to_output_dtype=False, gold_construction='independent_cpu_float64_scores', comparison='exact_int32_indices', comparison_dtype='int32')
        try:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), 'non-finite defined output or reference'
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False)
            record['status'] = 'passed'
        except AssertionError as error:
            record.update(status='precision_failed', error=str(error))
            failures.append(name)
        records.append(record)
    Path('cpu_fp64_accuracy.json').write_text(json.dumps(dict(validation_standard='cpu_fp64_topk_exact_indices_v1', accuracy_criteria_changed=True, original_validation_standard='cuda_fp16_kernel_integer_output_rtol_atol_1e-3', outputs=records), indent=2) + '\n')
    assert not failures, 'CPU FP64 TopK exact index validation failed: ' + '\n'.join((r.get('error', '') for r in records if r.get('status') == 'precision_failed'))
