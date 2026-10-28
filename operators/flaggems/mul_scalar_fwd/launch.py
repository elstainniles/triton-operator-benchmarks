"""Launch binding; input preparation and reference calculations are outside timing."""
import math


def grid(shape, config):
    return ((math.prod(shape) + config['B'] - 1) // config['B'],)


def launch(case, config):
    from . import kernels
    return kernels.mul_scalar_kernel[grid(case.inputs['shape'], config)](
        case.inputs['x'], case.actual[0], 0.75, case.inputs['shape'][0],
        str(case.dtype), config['B'], False, num_warps=32, num_stages=2)
