"""Liger catalog integrity and explicit extracted-kernel loading (no installs)."""
import ast
from collections import Counter
import hashlib
import importlib
import importlib.util
from pathlib import Path
import statistics
import sys

from .catalog import ROOT, operators, read_json, resolve

GROUP = ROOT / 'operators/liger_kernel'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_operator(path):
    meta = read_json(path/'operator.json')
    original = path/meta['source']['snapshot']
    assert digest(original) == meta['source']['sha256']
    assert digest(path/'kernels.py') == meta['execution']['kernel_sha256']
    assert digest(path/'test_case.py') == meta['correctness']['sha256']
    name = meta['kernel_symbol']
    functions = [n for n in ast.parse((path/'kernels.py').read_text(encoding='utf-8')).body
                 if isinstance(n, ast.FunctionDef)]
    assert len(functions) == 1 and functions[0].name == name
    source_function = next(n for n in ast.parse(original.read_text(encoding='utf-8')).body
                           if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(source_function) == ast.dump(functions[0]), ('Changed kernel/decorators', name)
    cases = read_json(path/'cases.json')['cases']
    assert len(cases) == 1 and cases[0]['nodeid'].startswith('test_case.py::')
    historical = read_json(path/'results/20261028/measurement.json')
    assert historical['kernel'] == name
    assert historical['source_sha256'] == meta['source']['sha256']
    if historical['status'] == 'measured':
        assert historical['correctness_passed'] and len(historical['samples_us']) == 15
        assert statistics.median(historical['samples_us']) == historical['median_us']
        assert all(x > 0 for x in historical['samples_us'])
    else:
        assert historical['median_us'] is None and historical['failure_reason']
    return meta, historical


def verify_suite():
    entries = [e for e in operators() if e.get('runner') == 'liger']
    assert len(entries) == 46
    scope = read_json(GROUP/'scope.json')
    assert {e['name'] for e in entries} == {r['name'] for r in scope['targets']}
    counts = Counter()
    for entry in entries:
        _, path = resolve(entry['id'])
        meta, history = verify_operator(path)
        selected = next(r['selected'] for r in scope['targets'] if r['name']==entry['name'])
        assert selected['path'] == meta['source']['path']
        assert selected['kind'] == entry['implementation']
        counts[history['status']] += 1
    assert dict(Counter(e['implementation'] for e in entries)) == scope['counts']
    manifest = read_json(GROUP/'source-manifest.json')
    for record in manifest['support']:
        assert digest(GROUP/record['path']) == record['sha256']
    assert dict(counts) == manifest['historical_counts']
    return dict(operators=len(entries), historical=dict(counts), kinds=scope['counts'])


def audit_dependency():
    """Check the wheel's complete Python payload before importing it."""
    spec = importlib.util.find_spec('liger_kernel')
    if spec is None or not spec.origin:
        raise RuntimeError('Pinned Liger wheel is required; see operators/liger_kernel/README.md')
    package = Path(spec.origin).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Liger must come from the isolated runtime wheel')
    manifest = read_json(GROUP/'dependency-manifest.json')
    for item in manifest['files']:
        path = package.parent/item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            raise RuntimeError('Pinned Liger dependency differs: '+item['path'])
    return dict(revision=manifest['revision'], package=str(package),
                verified_files=len(manifest['files']), manifest_sha256=digest(GROUP/'dependency-manifest.json'))


def load_kernel(path, meta):
    """Reuse upstream helpers/launcher, replacing only the selected JIT symbol."""
    module_name = meta['source']['path'].removeprefix('src/').removesuffix('.py').replace('/','.')
    module = importlib.import_module(module_name)
    if digest(module.__file__) != meta['source']['sha256']:
        raise RuntimeError('Selected installed module differs from frozen source')
    namespace = dict(vars(module))
    source = path/'kernels.py'
    exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), namespace)
    name = meta['kernel_symbol']
    setattr(module, name, namespace[name])
    # The upstream transformer tests import ops exports. Bind those exports to
    # the explicitly chosen generic/Ascend module before importing any tests.
    import liger_kernel.ops as ops
    rebound = []
    for symbol, value in vars(module).items():
        if getattr(value, '__module__', None) == module_name and not symbol.startswith('_'):
            setattr(ops, symbol, value)
            rebound.append(symbol)
    return dict(module=module_name, source=str(source), public_symbols=rebound,
                installed_source_sha256=digest(module.__file__), extracted_sha256=digest(source))
