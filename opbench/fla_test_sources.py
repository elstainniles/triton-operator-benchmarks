"""Device-free checks for FLA test sources and their in-memory fixtures."""
import ast
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

_PRIVATE = re.compile(r'/home/[A-Za-z0-9_.-]+/|/(?:data|workspace)/x\d+/'
                      r'|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'
                      r'|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def inside(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError('FLA test path escapes its directory: ' + str(name))
    if not path.is_file():
        raise ValueError('Missing FLA test source: ' + str(name))
    return path


def read_text(path):
    text = Path(path).read_text(encoding='utf-8')
    if _PRIVATE.search(text):
        raise ValueError('Personal path or credential in FLA test source: ' + str(path))
    return text


def read_json(path):
    return json.loads(read_text(path))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_python(path, provider=False):
    tree = ast.parse(read_text(path), filename=str(path))
    compile(tree, str(path), 'exec')  # Also rejects incorrectly placed future imports.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call = ast.unparse(node.func)
        if call in ('torch.load', 'torch.serialization.load'):
            raise ValueError('External tensor load remains in FLA test: ' + str(path))
        if provider and (call in ('torch.save', 'torch.serialization.save', 'open', 'eval', 'exec')
                         or call.startswith(('subprocess.', 'shutil.'))
                         or call.endswith(('.read_bytes', '.read_text', '.write_bytes', '.write_text',
                                           '.mkdir', '.open', '.load', '.save', '.exists', '.stat'))):
            raise ValueError('Fixture provider performs file I/O or dynamic execution: ' + call)
    return tree


def function_ast(node):
    node = copy.deepcopy(node)
    node.decorator_list = []
    class RemoveStringComments(ast.NodeTransformer):
        def visit_Expr(self, statement):
            if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                return None
            return self.generic_visit(statement)
    node = RemoveStringComments().visit(node)
    return ast.dump(node, include_attributes=False)


def verify_test(path, metadata):
    path = Path(path).resolve()
    record = read_json(path)
    directory = path.parent
    if (record['operator_id'] != metadata['id'] or record['kernel'] != metadata['kernel_symbol']
            or record['source_path'] != metadata['source']['path']):
        raise ValueError('FLA test kernel/source mapping differs from operator metadata')
    if record.get('external_input_required') is not False or record.get('input_origin') != 'seeded_cpu_generation':
        raise ValueError('FLA test must generate inputs without external tensor files')
    if not record['sides'] or not set(record['sides']) <= {'baseline', 'block'}:
        raise ValueError('Unknown or empty FLA test comparison side')
    files = {item['path']: item for item in record['files']}
    if len(files) != len(record['files']):
        raise ValueError('Duplicate FLA source file in test manifest')
    for key in ('entrypoint', 'provider'):
        if record[key] not in files:
            raise ValueError('FLA ' + key + ' is absent from source manifest')
    trees = {}
    for name, item in files.items():
        source = inside(directory, name)
        if digest(source) != item['sha256']:
            raise ValueError('FLA test source hash mismatch: ' + name)
        if source.suffix == '.py':
            trees[name] = audit_python(source, item['role'] == 'fixture_provider')
        elif source.suffix == '.json' and item['role'] == 'dependency':
            read_json(source)
        else:
            raise ValueError('FLA test source must be Python or JSON dependency metadata')
    if {name for name in files if name.endswith('.py')} != {str(p.relative_to(directory)) for p in directory.rglob('*.py')}:
        raise ValueError('FLA file manifest does not cover its Python sources')
    definitions = {n.name for tree in trees.values() for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if record['function'] not in definitions:
        raise ValueError('FLA test callable is absent from source closure')
    if 'make_fixtures' not in {n.name for n in trees[record['provider']].body if isinstance(n, ast.FunctionDef)}:
        raise ValueError('FLA fixture provider has no make_fixtures function')
    for relative in record['pythonpath']:
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory) or not candidate.is_dir():
            raise ValueError('FLA Python path escapes its directory or does not exist')
    bodies = [function_ast(n) for tree in trees.values() for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == record['kernel']]
    return dict(files=len(files), kernel_bodies=bodies, record=record)


def verify_suite(root=None):
    if root is None:
        from .catalog import ROOT
        root = ROOT
    root = Path(root).resolve()
    group = root / 'operators/flash_linear_attention'
    index = read_json(group / 'test-sources.json')
    if index.get('schema_version') != 1:
        raise ValueError('Unknown FLA test source index schema')
    catalog = {item['id']: item for item in read_json(root / 'catalog.json')['operators']
               if item.get('runner') == 'fla'}
    selectors, cases, paths, local_indexes, metadata_cache = set(), set(), set(), {}, {}
    counts = Counter()
    for item in index['tests']:
        selector = item['operator_id'], item['id']
        if selector in selectors:
            raise ValueError('Duplicate FLA test selector')
        selectors.add(selector)
        path = inside(root, item['path'])
        if path in paths:
            raise ValueError('Duplicate FLA test path')
        paths.add(path)
        operator_id = item['operator_id']
        if operator_id in catalog:
            operator_path = root / catalog[operator_id]['path']
            metadata_name = 'operator.json'
        elif operator_id.startswith('FLA_EXTRA_'):
            operator_path = path.parents[2]
            if operator_path.parent != group / 'additional_tests':
                raise ValueError('Supplemental FLA tests must stay in additional_tests')
            metadata_name = 'source-info.json'
        else:
            raise ValueError('Unknown FLA operator in test source index')
        if path != (operator_path / 'tests' / item['id'] / 'test.json').resolve():
            raise ValueError('FLA test index path differs from operator and selector')
        if operator_id not in metadata_cache:
            metadata = read_json(operator_path / metadata_name)
            if metadata['id'] != operator_id:
                raise ValueError('FLA operator metadata has a different ID')
            if metadata['source']['revision'] != index['source_revision']:
                raise ValueError('FLA source revision differs from global index')
            source = inside(operator_path, metadata['source']['snapshot'])
            if digest(source) != metadata['source']['sha256']:
                raise ValueError('FLA original source hash mismatch')
            original = ast.parse(source.read_text(encoding='utf-8'))
            bodies = {function_ast(n) for n in ast.walk(original) if isinstance(n, ast.FunctionDef)
                      and n.name == metadata['kernel_symbol']}
            metadata_cache[operator_id] = metadata, bodies
            local = read_json(operator_path / 'tests/manifest.json')
            if local['operator_id'] != operator_id or local['schema_version'] != 1:
                raise ValueError('FLA per-operator test manifest identity mismatch')
            local_map = {row['id']: row for row in local['tests']}
            if len(local_map) != len(local['tests']):
                raise ValueError('Duplicate selector in FLA operator test manifest')
            local_indexes[operator_id] = operator_path, local_map, set()
        metadata, original_bodies = metadata_cache[operator_id]
        _, local_map, used = local_indexes[operator_id]
        expected = dict(id=item['id'], case_id=item['case_id'], path=item['id'] + '/test.json')
        if local_map.get(item['id']) != expected:
            raise ValueError('FLA global and per-operator test mappings differ')
        used.add(item['id'])
        checked = verify_test(path, metadata)
        record = checked['record']
        if (record['id'] != item['id'] or record['case_id'] != item['case_id']
                or (root / record['path']).resolve() != path.parent):
            raise ValueError('FLA global and individual test mappings differ')
        cases.add(record['case_id'])
        counts['files'] += checked['files']
        if original_bodies and checked['kernel_bodies'] and not original_bodies.intersection(checked['kernel_bodies']):
            raise ValueError('FLA copied kernel body differs from the original source: ' + item['id'])
        # Keep the distinction visible: helper/decorator substitutions are not
        # proof that every copied kernel has the original complete AST.
        field = ('kernel_body_verified' if original_bodies.intersection(checked['kernel_bodies'])
                 else 'kernel_body_not_directly_comparable')
        counts[field] += 1
    for operator_path, local_map, used in local_indexes.values():
        if used != set(local_map):
            raise ValueError('FLA global index omits an operator test')
    manifests = {p.resolve() for pattern in ('*/tests/manifest.json', 'additional_tests/*/tests/manifest.json')
                 for p in group.glob(pattern)}
    if manifests != {(p / 'tests/manifest.json').resolve() for p, _, _ in local_indexes.values()}:
        raise ValueError('FLA global index omits an operator test manifest')
    actual = {p.resolve() for pattern in ('*/tests/*/test.json', 'additional_tests/*/tests/*/test.json')
              for p in group.glob(pattern)}
    if actual != paths:
        raise ValueError('FLA global index omits an individual test')
    summary = dict(operator_entries=len(local_indexes), test_variants=len(paths), case_ids=len(cases))
    for key, value in summary.items():
        if index['summary'].get(key) != value:
            raise ValueError('FLA test source summary mismatch: ' + key)
    return dict(summary, files=counts['files'], kernel_body_verified=counts['kernel_body_verified'],
                kernel_body_not_directly_comparable=counts['kernel_body_not_directly_comparable'])
