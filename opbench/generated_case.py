"""Generated fixtures with mixed-dtype checks and explicit in-place restoration.

Operator formulas stay in each operator's case.py. This module imports no
accelerator packages until a fixture is constructed by the audited runner.
"""
import math


class GeneratedCase:
    def __init__(self, torch, shape, dtype, device='npu'):
        self.torch, self.dtype, self.device = torch, dtype, device
        self.inputs = {'shape': tuple(shape)}
        self.actual, self.guards, self.checks = [], [], []
        self.readonly, self.restores = [], []

    def input(self, name, host):
        host = host.detach().contiguous().cpu()
        tensor = host.to(self.device, copy=True)
        self.inputs[name] = tensor
        self.readonly.append((name, tensor, host))
        return tensor

    def output(self, name, shape, dtype, expected, rtol, atol):
        length = math.prod(shape)
        storage = self.torch.full((length + 128,), 13.375, dtype=dtype, device=self.device)
        tensor = storage[64:-64].view(shape)
        self.actual.append(tensor)
        self.guards.append((name, storage))
        self.checks.append((name, tensor, expected.detach().cpu(), rtol, atol))
        return tensor

    def check(self, name, actual, expected, rtol, atol):
        """A callable actual is evaluated only after the kernel, outside timing."""
        self.checks.append((name, actual, expected.detach().cpu(), rtol, atol))

    def restore(self, target, source):
        self.restores.append((target, source))

    def prepare(self):
        # In-place outputs receive the original inputs, never a previous gradient.
        restored = {id(target) for target, _ in self.restores}
        for tensor in self.actual:
            if id(tensor) not in restored:
                tensor.fill_(float('nan'))
        for target, source in self.restores:
            target.copy_(source)

    def compare(self, name, actual, expected, rtol, atol):
        torch = self.torch
        if actual.shape != expected.shape:
            return dict(name=name, passed=False, reason='shape_mismatch')
        a, b = actual.detach().reshape(-1), expected.detach().reshape(-1)
        maximum, passed = 0.0, True
        # Bound temporary host/device memory even for the 2.5 GiB CE logits.
        for start in range(0, a.numel(), 1024 * 1024):
            x = a[start:start + 1024 * 1024].cpu().double()
            y = b[start:start + 1024 * 1024].cpu().double()
            finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            passed &= finite and bool(torch.isclose(x, y, rtol=rtol, atol=atol).all())
            if finite and maximum is not None:
                maximum = max(maximum, float((x - y).abs().max()))
            else:
                maximum = None
        return dict(name=name, passed=passed, elements=a.numel(), max_abs=maximum,
                    rtol=rtol, atol=atol)

    def validate(self):
        outputs = [self.compare(name, actual() if callable(actual) else actual, expected, rtol, atol)
                   for name, actual, expected, rtol, atol in self.checks]
        inputs = [self.compare(name, actual, expected, 0.0, 0.0)
                  for name, actual, expected in self.readonly]
        guards = all(bool((full[:64] == 13.375).all() and (full[-64:] == 13.375).all())
                     for _, full in self.guards)
        return dict(passed=bool(outputs) and all(c['passed'] for c in outputs + inputs) and guards,
                    outputs=outputs, inputs=inputs, guards=guards)
