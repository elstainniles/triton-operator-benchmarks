import torch_npu

from functools import lru_cache

@lru_cache
def is_arch35():
    return "Ascend910_95" in torch_npu.npu.get_device_name() or "Ascend950" in torch_npu.npu.get_device_name()
