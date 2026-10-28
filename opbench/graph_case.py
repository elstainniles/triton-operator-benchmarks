"""Generated inputs, CPU references and direct launches for the graph-test batch."""
import hashlib


class GraphCase:
    def __init__(self, torch, device='npu'):
        self.torch, self.device = torch, device
        self.inputs, self.cpu_inputs, self.outputs = {}, {}, []
        self.kwargs, self.grid, self.options = {}, (), {}
        self.accumulates = False

    def input(self, name, tensor):
        tensor = tensor.contiguous()
        self.cpu_inputs[name] = tensor
        self.inputs[name] = tensor.to(self.device)
        return self.inputs[name]

    def output(self, name, reference, rtol=2e-4, atol=1e-4, zero=False):
        torch = self.torch
        count = reference.numel()
        storage = torch.full((count + 32,), -16.0).to(reference.dtype)
        storage[16:16+count] = torch.full((count,), 0.0 if zero else float('nan')).to(reference.dtype)
        storage = storage.to(self.device)
        value = storage[16:16+count].view(reference.shape)
        self.outputs.append(dict(name=name, value=value, storage=storage, expected=reference,
                                 rtol=rtol, atol=atol, count=count))
        self.inputs[name] = value
        return value

    def prepare(self):
        # TG05 adds into its output. Reset outside the timed direct-launch label.
        if self.accumulates:
            self.outputs[0]['value'].zero_()

    def launch(self, kernels, symbol):
        options = dict(self.options, enable_graph_optimize=False)
        return getattr(kernels, symbol)[self.grid](**self.kwargs, **options)

    def validate(self):
        torch = self.torch
        checks, guards = [], True
        for output in self.outputs:
            actual = output['value'].detach().cpu().double()
            reference = output['expected'].double()
            finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
            delta = (actual - reference).abs()
            passed = finite and bool(torch.allclose(actual, reference, rtol=output['rtol'], atol=output['atol']))
            host = output['storage'].detach().cpu().float()
            guards &= bool((host[:16] == -16).all() and (host[-16:] == -16).all())
            checks.append(dict(name=output['name'], passed=passed, elements=actual.numel(),
                               max_abs=float(delta.max()) if finite else None,
                               rtol=output['rtol'], atol=output['atol']))
        return dict(passed=all(c['passed'] for c in checks) and guards, outputs=checks, guards=guards)

    def describe(self):
        return {name: dict(shape=list(t.shape), dtype=str(t.dtype).removeprefix('torch.'),
                           stride=list(t.stride()),
                           sha256=hashlib.sha256(t.view(self.torch.uint8).numpy().tobytes()).hexdigest())
                for name, t in self.cpu_inputs.items()}


def setup(config, device='npu'):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    case = GraphCase(torch, device)
    case.options = dict(config.get('launch_options', {'num_warps': 4, 'num_stages': 2}))
    return torch, generator, case


def ceildiv(a, b):
    return (a + b - 1) // b


def fp8_tensor(torch, tensor):
    # Quantize on CPU: no NPU preparation kernels enter the timed region.
    return tensor.clamp(-448, 448).to(torch.float8_e4m3fn)
