"""Launch binding; input preparation and reference calculations are outside timing."""
import math


def grid(shape, config):
    return ((math.prod(shape) + config['B'] - 1) // config['B'],)


def launch(case, config):
    from . import kernels
    m, n = case.inputs['shape']
    return kernels.mul_broadcast_2d_kernel[grid((m, n), config)](
        case.inputs['a'], case.inputs['b'], case.actual[0], m * n, n,
        1, 0, 0, 1, n, 1, str(case.dtype), config['B'], False,
        num_warps=32, num_stages=2)
