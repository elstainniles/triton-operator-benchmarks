#!/usr/bin/env python3
"""Port existing seeded FLA fixture branches to independent in-memory providers.

This development tool consumes a local test catalog. It never executes a kernel,
loads tensors, or publishes historical performance. Generated modules keep the
existing numerical statements and replace serialization with dictionary entries.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


class InMemory(ast.NodeTransformer):
    """Remove the old artifact bookkeeping, preserving tensor calculations."""
    def visit_Attribute(self, node):
        if ast.unparse(node) == 'args.output_dir':
            return ast.Call(func=ast.Name(id='PurePosixPath', ctx=ast.Load()),
                            args=[ast.Constant(value='')], keywords=[])
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            call = node.value
            name = ast.unparse(call.func)
            if name == 'torch.save':
                if len(call.args) != 2 or call.keywords:
                    raise ValueError('Unexpected source torch.save signature')
                path = self.visit(copy.deepcopy(call.args[1]))
                key = ast.Attribute(value=path, attr='name', ctx=ast.Load())
                return ast.Assign(targets=[ast.Subscript(
                    value=ast.Name(id='fixtures', ctx=ast.Load()), slice=key, ctx=ast.Store())],
                    value=self.visit(copy.deepcopy(call.args[0])))
            if name == 'artifacts.append' or name.endswith('.mkdir') or name.endswith('.write_text'):
                return None
        return self.generic_visit(node)

    def visit_For(self, node):
        node = self.generic_visit(node)
        return node if node.body else None


def constant_value(node, operation):
    names = {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}
    if not names <= {'selected', 'op'}:
        return None
    try:
        return eval(compile(ast.Expression(node), '<operation selector>', 'eval'),
                    {'__builtins__': {}}, {'selected': operation, 'op': operation})
    except (ValueError, TypeError, NameError, AttributeError):
        return None


class SelectOperation(ast.NodeTransformer):
    def __init__(self, operation):
        self.operation = operation

    def visit_If(self, node):
        value = constant_value(node.test, self.operation)
        if isinstance(value, bool):
            body = node.body if value else node.orelse
            output = []
            for part in body:
                transformed = self.visit(part)
                if isinstance(transformed, list):
                    output.extend(transformed)
                elif transformed is not None:
                    output.append(transformed)
            return output
        return self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == 'op':
            values = ast.literal_eval(node.iter)
            if self.operation not in values:
                return None
            output = []
            for part in node.body:
                transformed = self.visit(part)
                if isinstance(transformed, list):
                    output.extend(transformed)
                elif transformed is not None:
                    output.append(transformed)
            return output
        return self.generic_visit(node)


def port(source, operation):
    source = Path(source)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in tree.body if isinstance(node, ast.Assign)
                 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    start = next(index for index, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'selected' for t in node.targets))
    body = copy.deepcopy(main.body[start + 2:])  # Skip selected=args.operation and artifacts=[].
    function = ast.parse('def make_fixtures(seed=' + repr(constants['SEED']) + '):\n    pass').body[0]
    function.body = ast.parse(
        'import torch\nfrom pathlib import PurePosixPath\n'
        'SEED = seed\nN = ' + repr(constants['N']) + '\nselected = op = ' + repr(operation)
        + '\nfixtures = {}\n').body + body
    function = SelectOperation(operation).visit(function)
    function = InMemory().visit(function)
    function.body += ast.parse('return fixtures').body
    generated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    text = ('"""Seeded CPU inputs and independent references, ported from the existing FLA test generator.\n'
            'Logical fixture names are dictionary keys; this module reads and writes no data files.\n'
            'Source SHA256: ' + sha(source.read_bytes()) + '\nOperation: ' + operation + '\n"""\n'
            + ast.unparse(generated) + '\n')
    audit(text)
    return text


def audit(text):
    tree = ast.parse(text)
    forbidden = {'torch.load', 'torch.save', 'open', 'eval', 'exec', 'compile'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call = ast.unparse(node.func)
            if (call in forbidden or call.startswith(('subprocess.', 'os.', 'shutil.'))
                    or call.endswith(('.read_bytes', '.read_text', '.write_bytes', '.write_text',
                                      '.mkdir', '.open', '.stat', '.resolve', '.exists'))):
                raise ValueError('Unexpected I/O or dynamic execution in fixture: ' + call)
        if isinstance(node, ast.Name) and node.id in {'args', 'artifacts'}:
            raise ValueError('Unremoved generator bookkeeping: ' + node.id)
    compile(tree, '<in-memory FLA fixture>', 'exec')


def build(catalog, output):
    entries = read_json(catalog)['entries']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_cache, case_cache, records, providers = {}, {}, [], {}
    seen_paths = set()
    for entry in entries:
        case_path = entry.get('case_path')
        if not case_path or case_path in seen_paths or not Path(case_path).is_file():
            continue
        seen_paths.add(case_path)
        case = case_cache.setdefault(case_path, read_json(case_path))
        bundle = Path(entry['bundle_path'])
        manifest_file = bundle / 'synthetic/manifest.json'
        generator = bundle / 'synthetic/generate_data.py'
        if not manifest_file.is_file() or not generator.is_file():
            continue
        manifest = manifest_cache.setdefault(str(manifest_file), read_json(manifest_file))
        matches = [item for item in manifest['cases'] if item['id'] == case['id']
                   or item['original_nodeid'].split('::')[0] == case['original_source']]
        operations = {item['operation'] for item in matches}
        if len(operations) != 1:
            continue
        operation = operations.pop()
        source_hash = sha(generator.read_bytes())
        key = operation + '-' + source_hash[:12]
        if key not in providers:
            target = output / 'providers' / key / 'fixture.py'
            target.parent.mkdir(parents=True, exist_ok=True)
            text = port(generator, operation)
            target.write_text(text, encoding='utf-8')
            providers[key] = dict(operation=operation, source_sha256=source_hash,
                                  source_file=str(generator), fixture=str(target),
                                  fixture_sha256=sha(text.encode()), seed=20261030,
                                  ast_io_audit='passed', cpu_execution='not_run')
        records.append(dict(case_id=case['id'], case_path=case_path,
                            test_file=entry.get('test_file'), provider=key,
                            fixture=providers[key]['fixture'], operation=operation,
                            original_source=case['original_source']))
    result = dict(schema_version=1, purpose='in_memory_seeded_cpu_fixtures',
                  counts=dict(providers=len(providers), cases=len({r['case_id'] for r in records}),
                              case_paths=len(records)), providers=list(providers.values()), cases=records)
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    index = []
    for record in records:
        provider = providers[record['provider']]
        override = 'blue_adapters/grpo/' in str(record['test_file'])
        index.append(dict(case_path=record['case_path'], case_id=record['case_id'],
                          provider_dir=str(Path(record['fixture']).parent),
                          status='blocked' if override else 'ready',
                          reason=('This test uses a later GRPO input/reference schema; use its dedicated provider.'
                                  if override else None),
                          generator_provenance=dict(source=provider['source_file'],
                                                    sha256=provider['source_sha256'],
                                                    operation=record['operation']),
                          seed=provider['seed']))
    (output / 'index.json').write_text(json.dumps(dict(schema_version=1, entries=index),
                                                 ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.catalog, args.output)['counts']))


if __name__ == '__main__':
    main()
