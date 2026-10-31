"""Deterministic CPU inputs, FP64 reference and checked padded storage.

Only this directory owns the mathematical reference. The benchmark never needs
an external .pt file. CPU checks are not NPU execution evidence.
"""
import hashlib
import math

from opbench.graph_case import GraphCase


class GuardedCase(GraphCase):
    def __init__(self, torch, device):
        super().__init__(torch, device)
        self.input_storage = {}

    def padded_input(self, name, logical, padding=0):
        torch = self.torch
        rows, cols = logical.shape
        host = torch.full((rows * (cols + padding) + 32,), -16., dtype=logical.dtype)
        view = host[16:-16].view(rows, cols + padding)[:, :cols]
        view.copy_(logical)
        storage = host.to(self.device)
        value = storage[16:-16].view(rows, cols + padding)[:, :cols]
        self.cpu_inputs[name] = view
        self.inputs[name] = value
        self.input_storage[name] = (storage, host.clone())
        return value

    def padded_output(self, name, reference, padding=0):
        torch = self.torch
        rows, cols = reference.shape
        storage = torch.full((rows * (cols + padding) + 32,), -16., dtype=torch.float32,
                             device=self.device)
        value = storage[16:-16].view(rows, cols + padding)[:, :cols]
        value.fill_(float('nan'))
        self.outputs.append(dict(name=name, value=value, storage=storage, expected=reference,
                                 rtol=1e-5, atol=1e-5, count=reference.numel(),
                                 rows=rows, cols=cols, padding=padding))
        self.inputs[name] = value
        return value

    def prepare(self):
        # Reset outputs outside the timed launch. Inputs remain unchanged.
        for output in self.outputs:
            output['storage'].fill_(-16.)
            output['value'].fill_(float('nan'))

    def validate(self):
        torch = self.torch
        checks, guards = [], True
        for output in self.outputs:
            actual = output['value'].detach().cpu().double()
            reference = output['expected'].double()
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
            delta = (actual - reference).abs()
            maximum = float(delta.max()) if finite else None
            ratio = (float(delta.square().mean().sqrt()) /
                     (float(reference.square().mean().sqrt()) + 1e-8)) if finite else None
            # Original activations assert_close: max error <= 1e-6 OR RMS ratio < 1e-3.
            # Enforce this AND softmax's FP32 elementwise 1e-5/1e-5 gate for all six.
            upstream_gate = finite and (maximum <= 1e-6 or ratio < 1e-3)
            close = finite and bool(torch.allclose(actual, reference, rtol=1e-5, atol=1e-5))
            host = output['storage'].detach().cpu()
            padding = host[16:-16].view(output['rows'], output['cols'] + output['padding'])
            guard = bool((host[:16] == -16).all() and (host[-16:] == -16).all()
                         and (padding[:, output['cols']:] == -16).all())
            guards &= guard
            checks.append(dict(name=output['name'], passed=upstream_gate and close,
                               elements=actual.numel(), finite=finite, max_abs=maximum,
                               rms_ratio=ratio, rtol=1e-5, atol=1e-5,
                               upstream_activation_gate=upstream_gate, guards=guard))
        inputs_unchanged = all(torch.equal(storage.detach().cpu(), original)
                               for storage, original in self.input_storage.values())
        guards &= inputs_unchanged
        return dict(passed=all(c['passed'] for c in checks) and guards, outputs=checks,
                    guards=guards, inputs_unchanged=inputs_unchanged)

    def describe(self):
        return {name: dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix('torch.'),
                           stride=list(t.stride()),
                           sha256=hashlib.sha256(t.contiguous().view(self.torch.uint8).numpy().tobytes()).hexdigest(),
                           storage_sha256=hashlib.sha256(self.input_storage[name][1].view(self.torch.uint8).numpy().tobytes()).hexdigest())
                for name, t in self.cpu_inputs.items()}


def setup(config, device):
    import torch
    if config['dtype'] != 'float32':
        raise ValueError('These generated cases preserve the original FP32 acceptance only')
    rows, cols = config['shape']
    if not isinstance(rows, int) or not isinstance(cols, int) or rows <= 0 or cols <= 0:
        raise ValueError('shape must contain positive integer rows and columns')
    padding = config.get('row_padding', {})
    if any(not isinstance(p, int) or p < 0 for p in padding.values()):
        raise ValueError('row padding must be nonnegative integers')
    case = GuardedCase(torch, device)
    case.options = dict(config.get('launch_options', {'num_warps': 4, 'num_stages': 2}))
    if case.options.get('num_warps') not in (1, 2, 4, 8, 16, 32):
        raise ValueError('num_warps must be an original non-AMD autotune candidate')
    if case.options.get('num_stages') != 2:
        raise ValueError('num_stages is fixed at the original Config default, 2')
    if case.options.get('enable_graph_optimize', False):
        raise ValueError('graph optimization must remain disabled')
    rng = torch.Generator(device='cpu').manual_seed(config['seed'])
    return torch, rng, case, rows, cols, padding


def make_case(config, device='npu'):
    torch, rng, case, rows, cols, padding = setup(config, device)
    x_cpu = torch.randn(rows, cols, generator=rng, dtype=torch.float32)
    block = config.get('tile', {}).get('B', 512)
    if block not in (512, 1024, 2048, 4096, 8192):
        raise ValueError('B must be an original autotune candidate')
    case.padded_input('x', x_cpu, padding.get('x', 0))
    temperature = float(config.get('temperature', 1.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    dy_cpu = torch.randn(rows, cols, generator=rng, dtype=torch.float32)
    case.padded_input('dy', dy_cpu, padding.get('dy', 0))
    expected = dy_cpu.double() * torch.sigmoid(-x_cpu.double()) / temperature
    case.padded_output('dx', expected, padding.get('dx', 0))
    case.grid = ((rows * cols + block - 1) // block,)
    case.kwargs = dict(case.inputs, T=rows * cols, D=cols, B=block)
    for name, tensor in case.inputs.items():
        case.kwargs['stride_' + name + '_row'] = tensor.stride(0)
    case.kwargs['temperature'] = temperature
    return case
