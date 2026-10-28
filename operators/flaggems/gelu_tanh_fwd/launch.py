"""Launch binding; input preparation and reference calculations are outside timing."""
import math


def grid(shape, config):
    return ((math.prod(shape) + config['B'] - 1) // config['B'],)


def launch(case, config):
    from . import kernels
    return kernels.pointwise_probe[grid(case.inputs['shape'], config)](
        case.inputs['x'], case.inputs['y'], case.inputs['mask'], case.actual[0],
        case.inputs['shape'][0], config['B'], 1, kernels.gelu_tanh,
        num_warps=32, num_stages=2)
