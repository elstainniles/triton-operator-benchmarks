"""Original grid/tile; all setup is performed outside this direct launch."""

def grid(shape, config):
    return (shape[0],)


def launch(case, config):
    from .kernels import _rmsnorm_fwd_fused
    rows, cols = case.inputs['shape']
    return _rmsnorm_fwd_fused[grid((rows, cols), config)](
        case.inputs['x'], case.actual[0], case.inputs['weight'], case.actual[1],
        cols, cols, config['eps'], D_ROUNDED=1 << (cols - 1).bit_length())
