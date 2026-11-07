"""Original grid/tile; all setup is performed outside this direct launch."""

def grid(shape, config):
    return (shape[0], (shape[1] + config['BLOCK_SIZE'] - 1) // config['BLOCK_SIZE'])


def launch(case, config):
    from .kernels import _cross_entropy_bwd_kernel
    rows, cols = case.inputs['shape']
    # prepare() restored logits into this buffer outside the target timing label.
    target = case.actual[0]
    return _cross_entropy_bwd_kernel[grid((rows, cols), config)](
        target, target, case.inputs['row_max'], case.inputs['sum_exp'],
        case.inputs['labels'], case.inputs['dloss'], -1, cols, cols, cols,
        BLOCK_SIZE=config['BLOCK_SIZE'], STORE_BF16=config['STORE_BF16'])
