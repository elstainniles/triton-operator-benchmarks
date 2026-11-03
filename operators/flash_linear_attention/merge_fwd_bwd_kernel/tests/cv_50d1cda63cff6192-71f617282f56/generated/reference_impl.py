"""Existing independent CPU equations; file-generation main omitted."""
import torch

def reference_merge(data):
    a = data['ag_hm'].double()
    HV, K, V = (data['HV'], data['K'], data['V'])
    h = torch.zeros((HV, K, V), dtype=torch.float64)
    begin = data['rank'] - data['pre_num_ranks']
    for rank in range(begin, data['rank']):
        h = a[rank, :, :, V:] @ h + a[rank, :, :, :V]
    if data['transpose_state_layout']:
        h = h.transpose(-1, -2)
    result = h.unsqueeze(0).float()
    assert torch.isfinite(result).all()
    if data['pre_num_ranks'] == 1:
        expected = data['ag_hm'][begin, :, :, :V]
        if data['transpose_state_layout']:
            expected = expected.transpose(-1, -2)
        assert torch.equal(result[0], expected)
    return ({'initial_state': result}, {'method': 'CPU float64 affine-state recurrence across original preceding ranks', 'one_rank_identity_verified': data['pre_num_ranks'] == 1, 'original_acceptance': {'rtol': 1e-05, 'atol': 1e-05, 'equal_nan': True}})
