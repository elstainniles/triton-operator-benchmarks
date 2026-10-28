"""Offline audit of imported suites and the graph-disabled launch boundary."""
import ast
import hashlib

from .catalog import read_json


def launch_calls(tree, symbols):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
            and n.func.value.id in symbols]


def verify_graph_suite(path):
    metadata = read_json(path / 'operator.json')
    if metadata['compile_options'] != {'enable_graph_optimize': False}:
        raise ValueError('Graph optimization must be disabled for this batch')
    total = 0
    for record in metadata['files']:
        source = (path / record['snapshot']).read_bytes()
        execution = (path / record['executable']).read_bytes()
        for data, key in [(source, 'upstream_sha256'), (execution, 'executable_sha256')]:
            if hashlib.sha256(data).hexdigest() != record[key]:
                raise ValueError('Source hash mismatch: ' + record['executable'])
        git_blob = hashlib.sha1(b'blob ' + str(len(source)).encode() + b'\0' + source).hexdigest()
        if git_blob != record['git_blob']:
            raise ValueError('Upstream Git blob mismatch')
        original, actual = ast.parse(source), ast.parse(execution)
        calls = launch_calls(actual, metadata['kernel_symbols'])
        if len(calls) != record['graph_disabled_launches']:
            raise ValueError('Unexpected number of kernel launches')
        for call in calls:
            flags = [k for k in call.keywords if k.arg == 'enable_graph_optimize']
            if len(flags) != 1 or not isinstance(flags[0].value, ast.Constant) or flags[0].value.value is not False:
                raise ValueError('Kernel launch must explicitly set enable_graph_optimize=False')
            call.keywords.remove(flags[0])
        if ast.dump(actual) != ast.dump(original):
            raise ValueError('Imported executable changes more than graph optimization option')
        total += len(calls)
    if total != metadata['graph_disabled_launches'] or total == 0:
        raise ValueError('No audited graph-disabled launch')
    return dict(files=len(metadata['files']), graph_disabled_launches=total)
