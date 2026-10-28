"""Launch binding; input preparation and reference calculations are outside timing."""
import math


def grid(shape, config):
    return ((shape[0] + config['BM'] - 1) // config['BM'],
            (shape[1] + config['BH'] - 1) // config['BH'])


def launch(case, config):
    from . import kernels
    m, h = case.inputs['shape']
    return kernels.swiglu_kernel[grid((m, h), config)](
        case.inputs['ab'], case.actual[0], m, h, 2 * h, 1, h, 1,
        config['BM'], config['BH'], num_warps=32, num_stages=2)
