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

from collections import deque

from collections.abc import Callable

from typing import TYPE_CHECKING, Any

import torch

import triton

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'

FLA_TENSOR_CACHE_SIZE = int(os.getenv('FLA_TENSOR_CACHE_SIZE', "4"))

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters

autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

def tensor_cache(
    fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """
    A decorator that memoizes the most recent results of a function call by argument identity.

    The decorator keeps a bounded queue of up to ``FLA_TENSOR_CACHE_SIZE`` (default 4)
    recent ``(args, kwargs, result)`` triples. On each call, every cached entry is checked
    in order; an entry is considered a hit when the positional arg count and kwarg key set
    match and every argument is the *same object* (``is`` identity) as the cached one. On a
    hit the cached result is returned and ``fn`` is skipped; on a miss ``fn`` is invoked and
    the new triple is appended (evicting the oldest when the queue is full).

    Caching is fully bypassed when the ``FLA_DISABLE_TENSOR_CACHE`` environment variable is
    set to ``'1'``.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. Intended for functions whose inputs are tensors
            (or other objects compared by identity) and whose output is a tensor.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of ``fn`` backed by an identity-based bounded cache.
    """
    cached: deque = deque(maxlen=FLA_TENSOR_CACHE_SIZE)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if FLA_DISABLE_TENSOR_CACHE:
            return fn(*args, **kwargs)

        for cached_args, cached_kwargs, cached_result in cached:
            if len(args) != len(cached_args) or len(kwargs) != len(cached_kwargs):
                continue
            if all(a is b for a, b in zip(args, cached_args, strict=False)) and \
                    all(k in cached_kwargs and v is cached_kwargs[k] for k, v in kwargs.items()):
                return cached_result

        result = fn(*args, **kwargs)
        cached.append((args, kwargs, result))
        return result

    return wrapper

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

IS_AMD = (device_platform == 'hip')

IS_NVIDIA = (device_platform == 'cuda')

IS_NVIDIA_BLACKWELL = (IS_NVIDIA and torch.cuda.get_device_capability()[0] == 10)

USE_CUDA_GRAPH = (IS_NVIDIA and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

IS_TF32_SUPPORTED = (IS_NVIDIA and torch.cuda.get_device_capability(0)[0] >= 8)

IS_GATHER_SUPPORTED = hasattr(triton.language, 'gather')

if IS_NVIDIA and not IS_TF32_SUPPORTED:
    # Make old card happy, since triton will use tf32 by default.
    # This is a workaround for old nvidia card.
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
