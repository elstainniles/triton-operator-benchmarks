"""Original grid/tile; all setup is performed outside this direct launch."""

def grid(shape, config):
    return (shape[0], (shape[1] + config['BLOCK_SIZE'] - 1) // config['BLOCK_SIZE'])


def launch(case, config):
    from .kernels import _cross_entropy_fwd_kernel
    rows, cols = case.inputs['shape']
    return _cross_entropy_fwd_kernel[grid((rows, cols), config)](
        case.actual[0], case.actual[1], case.inputs['logits'], case.inputs['labels'],
        case.inputs['row_max'], -1, rows, cols, cols, BLOCK_SIZE=config['BLOCK_SIZE'])
