# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools

import inspect

import os

import warnings

import torch

import triton

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters

autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

def _cpu_device_warning():
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)

@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'

device_platform = get_available_device()

IS_NVIDIA = (device_platform == 'cuda')

USE_CUDA_GRAPH = (IS_NVIDIA and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

IS_TF32_SUPPORTED = (IS_NVIDIA and torch.cuda.get_device_capability(0)[0] >= 8)

if IS_NVIDIA and not IS_TF32_SUPPORTED:
    # Make old card happy, since triton will use tf32 by default.
    # This is a workaround for old nvidia card.
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
