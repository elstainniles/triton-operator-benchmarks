# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import contextlib

import functools

import inspect

import os

import warnings

from collections.abc import Callable

from enum import Enum

import torch

import triton

from packaging import version

FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters

autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

def input_guard(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    no_guard_contiguous: bool | list[str] = False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]] | Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.

    Args:
        no_guard_contiguous (bool | list[str]):
            If True, skip all contiguous checks. If a list of parameter names, skip contiguous check for those parameters.
    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        # Get function signature for parameter name mapping
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Convert no_guard_contiguous to list of parameter names if it's a list
            skip_params = set()
            if isinstance(no_guard_contiguous, list):
                skip_params = set(no_guard_contiguous)

            # Process args with parameter name mapping
            processed_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                else:
                    # For *args beyond signature, use position as name
                    param_name = f"__arg_{i}"

                if isinstance(arg, torch.Tensor):
                    if no_guard_contiguous is True or param_name in skip_params:
                        processed_args.append(arg)
                    else:
                        processed_args.append(arg.contiguous())
                else:
                    processed_args.append(arg)

            # Process kwargs
            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    if no_guard_contiguous is True or k in skip_params:
                        processed_kwargs[k] = v
                    else:
                        processed_kwargs[k] = v.contiguous()
                else:
                    processed_kwargs[k] = v

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    # Handle direct usage without parentheses: @input_guard
    if fn is not None:
        return decorator(fn)

    return decorator

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
