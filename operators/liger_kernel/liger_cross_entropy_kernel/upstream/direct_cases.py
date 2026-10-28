"""Correctness cases for exact JIT functions missed by upstream test selection.

The functions are imported from the registered wheel; kernel source is not copied.
"""

import importlib

import torch


def test_block_rms_norm():
    from liger_kernel.ops.rms_norm import rms_norm_backward, rms_norm_forward

    torch.manual_seed(20261028)
    x = torch.randn((32768, 128), device="npu", dtype=torch.float32)
    weight = torch.randn((128,), device="npu", dtype=torch.float32)
    dy = torch.randn_like(x)
    y, saved_x, rstd, block_size, warps, mode = rms_norm_forward(
        x, weight, 1e-6, 1.0, "gemma", False)
    dx, dw = rms_norm_backward(dy, saved_x, weight, rstd, 1.0, mode,
                               block_size, warps, False, False, needs_dw=True)
    ref_x = x.detach().requires_grad_(True)
    ref_w = weight.detach().requires_grad_(True)
    ref_y = ref_x * torch.rsqrt(ref_x.square().mean(-1, keepdim=True) + 1e-6) * (ref_w + 1.0)
    ref_dx, ref_dw = torch.autograd.grad(ref_y, (ref_x, ref_w), dy)
    torch.testing.assert_close(y, ref_y, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dx, ref_dx, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dw, ref_dw, rtol=1e-3, atol=1e-2)


def test_generic_softmax_multi_block_forward():
    from liger_kernel.ops.softmax import _softmax_multi_block_forward_kernel

    torch.manual_seed(20261028)
    x = torch.randn((16, 16384), device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    _softmax_multi_block_forward_kernel[(x.shape[0],)](
        y, y.stride(0), x, x.stride(0), x.shape[1], BLOCK_SIZE=1024, num_warps=4)
    ref_y = torch.softmax(x, dim=-1)
    torch.testing.assert_close(y, ref_y, rtol=1e-4, atol=1e-6)


def test_generic_softmax_multi_block_backward():
    from liger_kernel.ops.softmax import _softmax_multi_block_backward_kernel

    torch.manual_seed(20261028)
    x = torch.randn((16, 16384), device="npu", dtype=torch.float32)
    y = torch.softmax(x, dim=-1)
    dy = torch.randn_like(x)
    dx = torch.empty_like(x)
    _softmax_multi_block_backward_kernel[(x.shape[0],)](
        dy, dy.stride(0), y, y.stride(0), dx, dx.stride(0),
        x.shape[1], BLOCK_SIZE=1024, num_warps=4)
    ref_dx = y * (dy - (dy * y).sum(dim=-1, keepdim=True))
    torch.testing.assert_close(dx, ref_dx, rtol=1e-4, atol=1e-6)


def test_generic_selective_log_softmax():
    from liger_kernel.ops.grpo_loss import fused_selective_log_softmax

    torch.manual_seed(20261028)
    batch, length, vocab = 1, 1024, 64000
    temperature = 0.9
    logits = torch.randn((batch, length + 1, vocab), device="npu", dtype=torch.bfloat16)
    input_ids = torch.randint(0, vocab, (batch, length + 100), device="npu")
    actual = fused_selective_log_softmax(logits, input_ids, temperature)
    expected = torch.log_softmax(logits[:, :-1].float() / temperature, dim=-1)
    expected = expected.gather(-1, input_ids[:, -length:].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=0.5, atol=0.05)


def test_generic_layer_norm_backward():
    # Test the exact bwd independently of both backend dispatch and fwd compilation.
    from liger_kernel.ops.layer_norm import layer_norm_backward

    torch.manual_seed(20261028)
    x = torch.randn((4096, 4096), device="npu", dtype=torch.bfloat16)
    weight = torch.randn((4096,), device="npu", dtype=torch.bfloat16)
    bias = torch.randn_like(weight)
    dy = torch.randn_like(x)
    mean = x.float().mean(-1)
    rstd = torch.rsqrt((x.float() - mean[:, None]).square().mean(-1) + 1e-6)
    dx, dw, db = layer_norm_backward(dy, x, weight, bias, mean, rstd)
    ref_x, ref_w, ref_b = [value.detach().requires_grad_(True) for value in (x, weight, bias)]
    ref_y = torch.nn.functional.layer_norm(ref_x, (4096,), ref_w, ref_b, 1e-6)
    expected = torch.autograd.grad(ref_y, (ref_x, ref_w, ref_b), dy)
    # Same bfloat16 tolerance as the upstream LayerNorm correctness case.
    for actual, ref in zip((dx, dw, db), expected):
        torch.testing.assert_close(actual, ref, rtol=2e-2, atol=2e-2)


def test_generic_cross_entropy():
    # Retain the largest collected upstream shape/dtype and its numeric tolerances.
    # This invokes the installed generic wrapper, without the transformers registry.
    from liger_kernel.ops.cross_entropy import cross_entropy_forward

    torch.manual_seed(20261028)
    x = torch.randn((8192, 32000), device="npu", dtype=torch.bfloat16)
    labels = torch.randint(0, x.shape[1], (x.shape[0],), device="npu")
    ref_x = x.detach().clone().requires_grad_(True)
    ref_loss = torch.nn.functional.cross_entropy(ref_x, labels)
    ref_grad, = torch.autograd.grad(ref_loss, (ref_x,))
    actual_x = x.detach().clone().requires_grad_(True)
    loss, _, _, _, gradient = cross_entropy_forward(
        actual_x, labels, None, -100, 0.0, 0.0, "mean", None, False)
    torch.testing.assert_close(loss, ref_loss, rtol=0.05, atol=1e-8)
    torch.testing.assert_close(gradient, ref_grad, rtol=0.05, atol=1e-8)


def test_generic_element_mul():
    from liger_kernel.ops.utils import element_mul_kernel

    torch.manual_seed(20261028)
    x = torch.randn((8192, 4096), device="npu", dtype=torch.float32)
    scalar = torch.tensor(-1.25, device="npu", dtype=torch.float32)
    expected = x * scalar
    element_mul_kernel[(x.shape[0],)](x, x.stride(0), scalar, x.shape[1],
                                    BLOCK_SIZE=4096, num_warps=4)
    torch.testing.assert_close(x, expected, rtol=1e-6, atol=1e-6)


def _embedding_inputs():
    embedding = importlib.import_module("liger_kernel.ops.backends._ascend.ops.embedding")
    torch.manual_seed(20261028)
    weight = torch.randn((8192, 1024), device="npu", dtype=torch.float32)
    indices = torch.randint(0, weight.shape[0], (2048,), device="npu")
    return embedding, weight, indices


def test_ascend_embedding_forward():
    embedding, weight, indices = _embedding_inputs()
    output = embedding.embedding_forward(weight, indices)
    expected = torch.nn.functional.embedding(indices, weight)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_ascend_embedding_backward():
    embedding, weight, indices = _embedding_inputs()
    torch.manual_seed(20261028)
    output = torch.nn.functional.embedding(indices, weight)
    dy = torch.randn_like(output)
    dw = embedding.embedding_backward(weight, indices, dy)
    expected_dw = torch.zeros_like(weight)
    expected_dw.index_add_(0, indices, dy)
    torch.testing.assert_close(dw, expected_dw, rtol=1e-3, atol=1e-3)


def test_experimental_int8int2_matmul():
    from liger_kernel.ops.experimental.mm_int8int2 import matmul, unpack_weights

    torch.manual_seed(20261028)
    a = torch.randint(-2, 3, (256, 2048), device="npu", dtype=torch.int8)
    packed_b = torch.randint(0, 256, (512, 512), device="npu", dtype=torch.uint8)
    actual = matmul(a, packed_b)
    unpacked_b = unpack_weights(packed_b)
    expected = a.to(torch.float32) @ unpacked_b.to(torch.float32)
    torch.testing.assert_close(actual, expected.to(torch.int32), rtol=0, atol=0)
