"""Original grid/tile; all setup is performed outside this direct launch."""

def grid(shape, config):
    return (config['GROUPS'],)


def launch(case, config):
    from .kernels import _rmsnorm_bwd_dxdw_fused
    rows, cols = case.inputs['shape']
    return _rmsnorm_bwd_dxdw_fused[grid((rows, cols), config)](
        case.inputs['dy'], case.actual[0], case.actual[1], case.inputs['x'],
        case.inputs['weight'], case.inputs['rrms'], cols, rows, cols,
        BLOCK_S=(rows + config['GROUPS'] - 1) // config['GROUPS'],
        D_ROUNDED=1 << (cols - 1).bit_length())
