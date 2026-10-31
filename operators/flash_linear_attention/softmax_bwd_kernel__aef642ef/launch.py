"""Direct target-kernel binding; preparation/reference are outside timing."""


def launch(case, config=None):
    from . import kernels
    config = dict(config or {})
    kwargs = dict(case.kwargs)
    options = dict(case.options)
    if 'B' in config:
        kwargs['B'] = config.pop('B')
    options.update(config)
    block = kwargs['B']
    if block != 1 << (kwargs['D'] - 1).bit_length():
        raise ValueError('softmax keeps original next_power_of_2(D)')
    grid = case.grid
    options['enable_graph_optimize'] = False
    return kernels.softmax_bwd_kernel[grid](**kwargs, **options)
