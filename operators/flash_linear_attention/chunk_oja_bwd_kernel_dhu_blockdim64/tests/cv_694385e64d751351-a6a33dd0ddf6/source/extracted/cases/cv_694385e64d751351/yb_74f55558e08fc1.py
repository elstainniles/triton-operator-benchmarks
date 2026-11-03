# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools

import inspect

import os

import sys

import warnings

from collections import deque

from collections.abc import Callable

from enum import Enum

from typing import TYPE_CHECKING, Any

import torch

import triton

from packaging import version

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

@functools.cache
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)

def _cpu_device_warning():
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)

@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'

device = get_available_device() if get_available_device() != 'hip' else 'cuda'

device_torch_lib = getattr(torch, device)

device_platform = get_available_device()

IS_NVIDIA = (device_platform == 'cuda')

IS_NVIDIA_HOPPER = (IS_NVIDIA and ('NVIDIA H' in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] == 9))

USE_CUDA_GRAPH = (IS_NVIDIA and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

IS_TF32_SUPPORTED = (IS_NVIDIA and torch.cuda.get_device_capability(0)[0] >= 8)

if IS_NVIDIA and not IS_TF32_SUPPORTED:
    # Make old card happy, since triton will use tf32 by default.
    # This is a workaround for old nvidia card.
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]

class Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value

@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False

if check_pytorch_version('2.4'):
    device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == 'cuda', 'Only cuda device is supported for PyTorch version < 2.4.0.'
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)

def _register_aliases():
    current_module = sys.modules[__name__]
    for key in (
        'IS_AMD',
        'IS_ARM',
        'IS_INTEL',
        'IS_INTEL_ALCHEMIST',
        'IS_NVIDIA',
        'IS_NVIDIA_BLACKWELL',
        'IS_NVIDIA_HOPPER',
        'USE_CUDA_GRAPH',
        'IS_TF32_SUPPORTED',
        'IS_GATHER_SUPPORTED',
        'IS_TMA_SUPPORTED',
    ):
        if hasattr(current_module, key):
            setattr(current_module, key.lower(), getattr(current_module, key))

_register_aliases()
