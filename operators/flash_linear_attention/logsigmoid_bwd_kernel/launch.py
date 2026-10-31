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
    if block not in (512, 1024, 2048, 4096, 8192):
        raise ValueError('B must be an original autotune candidate')
    grid = ((kwargs['T'] + block - 1) // block,)
    options['enable_graph_optimize'] = False
    return kernels.logsigmoid_bwd_kernel[grid](**kwargs, **options)
