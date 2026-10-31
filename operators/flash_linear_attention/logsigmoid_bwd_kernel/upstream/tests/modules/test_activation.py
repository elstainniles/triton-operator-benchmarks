# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules.activations import logsigmoid, sigmoid, swiglu, swiglu_linear, swish
from fla.utils import assert_close, device


def _get_device_type():
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


_device_type = _get_device_type()


def _profile(fn_triton):
    if _device_type == 'npu':
        import torch_npu
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False
        )
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=10, repeat=1, skip_first=1),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir")
        ) as prof:
            for i in range(13):
                fn_triton()
                torch.npu.synchronize()
                prof.step()
    elif _device_type == 'cuda':
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1, skip_first=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./result_dir")
        ) as prof:
            for i in range(13):
                fn_triton()
                torch.cuda.synchronize()
                prof.step()
        print("\n========== GPU 性能统计 ==========")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    else:
        pytest.skip("未检测到 GPU 或 NPU，跳过性能 profiling")
    print("性能 profiling 完成！")


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 1024, 512, False),
        (2, 1536, 768, False),
        (2, 1024, 1200, True),
        (3, 2048, 1200, True),
        (4, 2048, 1200, False),
        (4, 1024, 512, False),
        (3, 2048, 1024, False),
        (3, 1536, 15636, True),
        (5, 1024, 4096, False),
        (4, 1024, 5120, True),
    ],
)
def test_sigmoid(B: int, T: int, D: int, compile: bool):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y_ref = torch.sigmoid(x)
    y_tri = sigmoid(x) if not compile else torch.compile(sigmoid)(x)

    g = torch.randn_like(y_ref)
    dx_ref = torch.autograd.grad(y_ref, x, g)[0]
    dx_tri = torch.autograd.grad(y_tri, x, g)[0]

    assert_close('sigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('sigmoid bwd', dx_ref, dx_tri, 1e-3)

    # if not compile:
    _profile(lambda: torch.autograd.grad(sigmoid(x), x, g))


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'temperature', 'compile'),
    [
        (2, 1024, 512, 1.0,  False),
        (2, 2048, 512, 0.5,  False),
        (2, 512, 1024, 0.5,  True),
        (3, 2048, 1200, 2.0, True),
        (4, 2048, 1500, 2.0, True),
        (3, 1024, 1200, 0.75, True),
        (3, 1536, 1536, 0.5, False),
        (4, 1024, 2048, 1.0, True),
        (5, 1024, 2048, 0.75, False),
        (5, 1024, 1024, 0.5, True),
    ],
)
def test_logsigmoid(B: int, T: int, D: int, temperature: float, compile: bool):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y_ref = F.logsigmoid(x) / temperature
    y_tri = logsigmoid(x, temperature) if not compile else torch.compile(logsigmoid)(x, temperature)

    g = torch.randn_like(y_ref)
    dx_ref = torch.autograd.grad(y_ref, x, g)[0]
    dx_tri = torch.autograd.grad(y_tri, x, g)[0]

    assert_close('logsigmoid fwd', y_ref, y_tri, 1e-3)
    assert_close('logsigmoid bwd', dx_ref, dx_tri, 1e-3)

    #if not compile:
    _profile(lambda: torch.autograd.grad(logsigmoid(x, temperature), x, g))


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 1024, 512, False),
        (2, 1536, 768, False),
        (2, 2048, 1024, True),
        (3, 2048, 1024, True),
        (4, 2048, 1200, False),
        (3, 1024, 1200, False),
        (3, 1536, 1536, True),
        (4, 1280, 2560, True),
        (4, 1280, 2560, True),
        (5, 1024, 2048, True),
    ],
)
def test_swish(B: int, T: int, D: int, compile: bool):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y_ref = F.silu(x)
    y_tri = swish(x) if not compile else torch.compile(swish)(x)

    g = torch.randn_like(y_ref)
    dx_ref = torch.autograd.grad(y_ref, x, g)[0]
    dx_tri = torch.autograd.grad(y_tri, x, g)[0]

    assert_close('swish fwd', y_ref, y_tri, 1e-3)
    assert_close('swish bwd', dx_ref, dx_tri, 1e-3)

    #if not compile:
    _profile(lambda: torch.autograd.grad(swish(x), x, g))


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'compile'),
    [
        (2, 1024, 512, False),
        (2, 2048, 512, False),
        (2, 512, 1024, True),
        (3, 2048, 1200, True),
        (4, 2048, 1500, True),
        (3, 1024, 1200, True),
        (3, 1536, 1536, False),
        (4, 1024, 2048, True),
        (5, 1024, 2048, False),
        (5, 1024, 1024, True),
    ],
)
def test_swiglu(B: int, T: int, D: int, compile: bool):
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y = torch.randn(B, T, D, device=device, requires_grad=True)

    y_ref = F.silu(x) * y
    y_tri = swiglu(x, y) if not compile else torch.compile(swiglu)(x, y)

    g = torch.randn_like(y_ref)
    dx_ref, dy_ref = torch.autograd.grad(y_ref, (x, y), g)
    dx_tri, dy_tri = torch.autograd.grad(y_tri, (x, y), g)

    assert_close('swiglu fwd', y_ref, y_tri, 1e-3)
    assert_close('swiglu dx',  dx_ref, dx_tri, 1e-3)
    assert_close('swiglu dy',  dy_ref, dy_tri, 1e-3)


@pytest.mark.parametrize(
    ('B', 'T', 'D', 'O', 'compile'),
    [
        (2, 512, 128, 256, True),
        (1, 1, 64, 32, False),
        (2, 500, 128, 64, True),
        (3, 2048, 1200, 600, False),
    ],
)
def test_swiglu_linear(B: int, T: int, D: int, O: int, compile: bool):  # noqa: E741
    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    y = torch.randn(B, T, D, device=device, requires_grad=True)
    w = torch.randn(O, D, device=device, requires_grad=True)
    b = torch.randn(O, device=device, requires_grad=True)

    z_ref = F.silu(x) * y
    out_ref = F.linear(z_ref, w, b)
    out_tri = swiglu_linear(x, y, w, b) if not compile else torch.compile(swiglu_linear)(x, y, w, b)

    g = torch.randn_like(out_ref)
    dx_ref, dy_ref, dw_ref, db_ref = torch.autograd.grad(out_ref, (x, y, w, b), g)
    dx_tri, dy_tri, dw_tri, db_tri = torch.autograd.grad(out_tri, (x, y, w, b), g)

    assert_close('swiglu_linear out', out_ref, out_tri, 1e-3)
    assert_close('swiglu_linear dx',  dx_ref,  dx_tri,  1e-3)
    assert_close('swiglu_linear dy',  dy_ref,  dy_tri,  1e-3)
    assert_close('swiglu_linear dw',  dw_ref,  dw_tri,  1e-3)
    assert_close('swiglu_linear db',  db_ref,  db_tri,  1e-3)
