"""Original grid/tile; all setup is performed outside this direct launch."""

def grid(shape, config):
    return (shape[0], (shape[1] + config['BLOCK'] - 1) // config['BLOCK'])


def launch(case, config):
    from .kernels import _fused_situgluv2_fwd_triton_kernel
    rows, cols = case.inputs['shape']
    return _fused_situgluv2_fwd_triton_kernel[grid((rows, cols), config)](
        case.inputs['a'], case.inputs['b'], case.actual[0], cols,
        cols, 1, cols, 1, cols, 1, 0.5, 2.0, 0.5, 2.0, BLOCK=config['BLOCK'])
