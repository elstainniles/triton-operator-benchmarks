"""Full-output checks and output sentinels shared by independent operator cases."""
import math


def random_cpu(torch, shape, dtype):
    return torch.randn(shape, dtype=torch.float32).to(dtype)


class Case:
    def __init__(self, torch, dtype):
        self.torch = torch
        self.dtype = dtype
        self.inputs = {}
        self.actual = []
        self.expected = []
        self.guards = []

    def output(self, shape):
        length = math.prod(shape)
        storage = self.torch.full((length + 32,), -123, dtype=self.dtype, device='npu')
        output = storage[16:16 + length].view(shape)
        self.guards.append((storage, length))
        return output

    def validate(self):
        torch = self.torch
        rtol, atol = (2e-2, 2e-2) if self.dtype == torch.bfloat16 else (2e-4, 2e-5)
        checks = []
        for actual, expected in zip(self.actual, self.expected, strict=True):
            actual, expected = actual.detach().cpu().double(), expected.double()
            finite = torch.isfinite(actual) & torch.isfinite(expected)
            difference = (actual - expected).abs()
            checks.append(dict(passed=bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)),
                               elements=actual.numel(), rtol=rtol, atol=atol,
                               max_abs=float(difference[finite].max()) if finite.any() else None))
        guards = True
        for storage, length in self.guards:
            host = storage.cpu()
            guards &= bool((host[:16] == -123).all() and (host[16 + length:] == -123).all())
        return dict(passed=bool(checks) and all(c['passed'] for c in checks) and guards,
                    outputs=checks, guards=guards)
