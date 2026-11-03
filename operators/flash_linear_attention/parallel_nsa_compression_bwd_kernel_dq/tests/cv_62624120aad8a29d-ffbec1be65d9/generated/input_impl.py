from input_manual import build

def make_inputs():
    result = build('parallel_nsa_compression_bwd_kernel_dq')
    grid = None
    if isinstance(result, tuple):
        data = result[0]
        if len(result) == 4:
            grid = result[3]
    else:
        data = result
    if 'input_data' in data:
        return {'data': data, 'grid': data.get('grid', grid)}
    return {'inputs_dict': data, 'grid': grid}
