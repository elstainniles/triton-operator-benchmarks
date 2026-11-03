"""Existing independent CPU equations; file-generation main omitted."""
import torch

def make_inputs():
    torch.manual_seed(20261031)
    HV, K, V = (8, 64, 128)
    num_ranks = 2
    rank = 1
    device = torch.device('cpu')
    dtype = torch.float32
    ag_hm = torch.randn(num_ranks, HV, K, K + V, dtype=dtype, device=device)
    data = {}
    data['input_data'] = {}
    data['input_data']['ag_hm'] = ag_hm.cpu()
    data['input_data']['HV'] = HV
    data['input_data']['K'] = K
    data['input_data']['V'] = V
    data['input_data']['pre_num_ranks'] = rank
    data['input_data']['rank'] = rank
    data['input_data']['chunk_size'] = 64
    data['input_data']['transpose_state_layout'] = False
    _prefix_values = locals().copy()
    if not (any((isinstance(_prefix_values.copy().get(key), dict) and _prefix_values.copy().get(key) for key in ('inputs_dict', 'input_dict', 'input_data'))) or (isinstance(_prefix_values.copy().get('data'), dict) and _prefix_values.copy()['data'].get('input_data'))):
        try:
            data = {}
        except NameError:
            pass
        try:
            data['input_data'] = {}
        except NameError:
            pass
        try:
            data['input_data']['ag_hm'] = ag_hm.cpu()
        except NameError:
            pass
        try:
            data['input_data']['HV'] = HV
        except NameError:
            pass
        try:
            data['input_data']['K'] = K
        except NameError:
            pass
        try:
            data['input_data']['V'] = V
        except NameError:
            pass
        try:
            data['input_data']['pre_num_ranks'] = rank
        except NameError:
            pass
        try:
            data['input_data']['rank'] = rank
        except NameError:
            pass
        try:
            data['input_data']['chunk_size'] = 64
        except NameError:
            pass
        try:
            data['input_data']['transpose_state_layout'] = False
        except NameError:
            pass
    return locals()
