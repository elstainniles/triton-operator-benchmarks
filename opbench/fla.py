"""Offline FLA source audits and plans; archives never import accelerator code."""
import ast
import hashlib
import re

from .catalog import read_json


def local_file(path, relative):
    candidate = (path / relative).resolve()
    if not candidate.is_relative_to(path.resolve()) or not candidate.is_file():
        raise ValueError('FLA file must exist inside its operator directory: ' + relative)
    return candidate


def function(path, symbol):
    matches = [node for node in ast.parse(path.read_bytes()).body
               if isinstance(node, ast.FunctionDef) and node.name == symbol]
    if len(matches) != 1:
        raise ValueError('Expected exactly one FLA source function: ' + symbol)
    return matches[0]


def without_autotune(node):
    node.decorator_list = [decorator for decorator in node.decorator_list
                           if not (isinstance(decorator, ast.Call)
                                   and isinstance(decorator.func, ast.Attribute)
                                   and isinstance(decorator.func.value, ast.Name)
                                   and decorator.func.value.id == 'triton'
                                   and decorator.func.attr == 'autotune')]
    return node


def helper_function(path, symbol, policy):
    if policy != 'default_FLA_USE_FAST_OPS_0':
        return function(path, symbol)
    expected = ast.dump(ast.parse("os.environ.get('FLA_USE_FAST_OPS', '0') == '1'", mode='eval').body)
    branches = [node for node in ast.parse(path.read_bytes()).body
                if isinstance(node, ast.If) and ast.dump(node.test) == expected]
    if len(branches) != 1:
        raise ValueError('Expected the original FLA_USE_FAST_OPS branch')
    matches = [node for node in branches[0].orelse
               if isinstance(node, ast.FunctionDef) and node.name == symbol]
    if len(matches) != 1:
        raise ValueError('Expected one default FLA helper: ' + symbol)
    return matches[0]


def verify_operator(path):
    meta = read_json(path / 'operator.json')
    if meta.get('runner') != 'fla' or meta.get('entry_mode') not in (
            'source_archive', 'generated_benchmark'):
        raise ValueError('Unknown FLA runner or entry mode')
    source, extraction = meta['source'], meta['extraction']
    if not re.fullmatch('[0-9a-f]{40}', source['revision']):
        raise ValueError('FLA source must pin a full Git revision')
    for record, filename in ((source, source['snapshot']), (extraction, extraction['path'])):
        actual = hashlib.sha256(local_file(path, filename).read_bytes()).hexdigest()
        if actual != record['sha256']:
            raise ValueError('FLA source hash mismatch: ' + filename)
    dependency_paths = set()
    for dependency in meta.get('dependencies', []):
        filename = dependency['path']
        if filename in dependency_paths:
            raise ValueError('Duplicate FLA dependency path: ' + filename)
        dependency_paths.add(filename)
        actual = hashlib.sha256(local_file(path, filename).read_bytes()).hexdigest()
        if actual != dependency['sha256']:
            raise ValueError('FLA dependency hash mismatch: ' + filename)
    if source['symbol'] != meta['kernel_symbol'] or extraction.get('ast_preserved') is not True:
        raise ValueError('FLA extraction must preserve its original symbol and AST')
    original = function(local_file(path, source['snapshot']), source['symbol'])
    extracted = function(local_file(path, extraction['path']), meta['kernel_symbol'])
    policy = meta.get('decorator_policy', 'preserve')
    if policy == 'fixed_autotune_candidate':
        original = without_autotune(original)
    elif policy != 'preserve':
        raise ValueError('Unknown FLA decorator policy')
    if ast.dump(original) != ast.dump(extracted):
        raise ValueError('FLA extraction changes original signature, decorators, or body')
    for helper in meta.get('kernel_helpers', []):
        snapshot = local_file(path, helper['snapshot'])
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != helper['sha256']:
            raise ValueError('FLA helper source hash mismatch: ' + helper['symbol'])
        if ast.dump(helper_function(snapshot, helper['symbol'], meta.get('helper_policy'))) != ast.dump(
                function(local_file(path, extraction['path']), helper['symbol'])):
            raise ValueError('FLA extraction changes helper: ' + helper['symbol'])
    cases = read_json(path / 'cases.json')
    if cases.get('external_input_required') is not False:
        raise ValueError('FLA entries cannot require unpublished inputs')
    if meta['entry_mode'] == 'source_archive':
        if cases['cases'] or not meta.get('adaptation_reason'):
            raise ValueError('FLA archives require an explicit reason and no runnable cases')
    else:
        if not cases['cases']:
            raise ValueError('Generated FLA entries require explicit input cases')
        local_file(path, 'benchmark.py')
        ids = [case['id'] for case in cases['cases']]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate FLA case ID')
        for case in cases['cases']:
            if not isinstance(case.get('seed'), int):
                raise ValueError('Generated FLA cases require a fixed integer seed')
            for key in ('shape', 'dtype', 'suite'):
                if key not in case:
                    raise ValueError('Missing FLA case field: ' + key)
    return meta, cases


def plans(entry, path, suite, variants):
    if variants == 'manual':
        raise ValueError('FLA entries provide original baselines; no manual candidate is registered')
    meta, matrix = verify_operator(path)
    for key in ('id', 'runner', 'entry_mode'):
        if meta[key] != entry[key]:
            raise ValueError('FLA catalog and metadata disagree: ' + key)
    common = dict(operator_id=entry['id'], operator=entry['name'], variant_id='baseline',
                  runner='tools/run.py run', source_kind=meta['entry_mode'],
                  required_input_file=None)
    if meta['entry_mode'] == 'source_archive':
        return [dict(common, case=None, status='not_executable',
                     reason=meta['adaptation_reason'])]
    dtype = {'fp32': 'float32', 'bf16': 'bfloat16'}.get(suite)
    return [dict(common, case=case, status='ready',
                 compile_options={'enable_graph_optimize': False})
            for case in matrix['cases']
            if suite == 'all' or (suite == 'smoke' and case['suite'] == 'smoke')
            or (dtype is not None and case['dtype'] == dtype)]


def verify_scope(entries, metadata, manifest, scope, inventory):
    """Require the entire imported launch inventory, including blocked entries."""
    def unique(records, key, label):
        values = [key(record) for record in records]
        if len(set(values)) != len(values):
            raise ValueError('Duplicate FLA ' + label)
        return set(values)

    catalog_ids = unique(entries, lambda row: row['id'], 'catalog ID')
    manifest_ids = unique(manifest['operators'], lambda row: row['id'], 'manifest ID')
    metadata_ids = unique(metadata, lambda row: row['id'], 'metadata ID')
    if catalog_ids != manifest_ids or catalog_ids != metadata_ids:
        raise ValueError('FLA catalog does not cover the full source manifest')
    unique(entries, lambda row: row['path'], 'catalog path')
    key = lambda row: (row['source']['path'], row['kernel_symbol'])
    actual_keys = unique(metadata, key, 'source path and symbol')
    manifest_keys = unique(manifest['operators'], key, 'manifest source path and symbol')
    definitions = inventory['kernel_definitions']
    launched = [row for row in definitions if row['static_launch_references']]
    inventory_keys = unique(launched, lambda row: (row['source'], row['kernel']), 'inventory source path and symbol')
    if actual_keys != manifest_keys or actual_keys != inventory_keys:
        raise ValueError('FLA entries do not cover the full static launch inventory')
    if (len(definitions) != scope['static_jit_definitions']
            or len(launched) != scope['launch_referenced_operator_entries']
            or len(definitions) - len(launched) != scope['jit_definitions_without_static_launch_reference']):
        raise ValueError('FLA scope counts disagree with the imported inventory')
    revision = scope['source_revision']
    if manifest['source_revision'] != revision or inventory['source_commit'] != revision:
        raise ValueError('FLA manifests disagree on the source revision')
    original = {row['id']: row for row in manifest['operators']}
    catalog = {row['id']: row for row in entries}
    inventory_by_key = {(row['source'], row['kernel']): row for row in launched}
    for meta in metadata:
        if key(meta) != key(original[meta['id']]):
            raise ValueError('FLA operator ID changed its source identity')
        for field in ('runner', 'entry_mode'):
            if meta[field] != catalog[meta['id']][field]:
                raise ValueError('FLA catalog and metadata disagree: ' + field)
        source, archived = meta['source'], original[meta['id']]['source']
        for field in ('revision', 'repository', 'path', 'symbol', 'sha256', 'snapshot'):
            if source[field] != archived[field]:
                raise ValueError('FLA source identity differs from its import manifest: ' + field)
        if source['revision'] != revision or source['repository'] != scope['source_repository']:
            raise ValueError('FLA operator has a different source checkout')
        definition = inventory_by_key[key(meta)]
        if source.get('original_sha256', source['sha256']) != definition['original_source_sha256']:
            raise ValueError('FLA original source hash differs from the import inventory')
        if meta['static_launch_references'] != definition['static_launch_references']:
            raise ValueError('FLA static launch references changed')


def verify_suite():
    from .catalog import ROOT, resolve, select
    entries = select('fla')
    group = ROOT / 'operators/flash_linear_attention'
    if not entries and not group.exists():
        return dict(entries=0, source_archive=0, generated_benchmark=0, generated_cases=0)
    manifest = read_json(group / 'source-manifest.json')
    scope = read_json(group / 'scope.json')
    inventory = read_json(group / 'import-manifest.json')['inventory']
    if hashlib.sha256((group / 'LICENSE').read_bytes()).hexdigest() != manifest['license_sha256']:
        raise ValueError('FLA license hash mismatch')
    counts = dict(entries=0, source_archive=0, generated_benchmark=0, generated_cases=0)
    metadata = []
    for entry in entries:
        _, path = resolve(entry['id'])
        meta, cases = verify_operator(path)
        metadata.append(meta)
        counts['entries'] += 1
        counts[meta['entry_mode']] += 1
        counts['generated_cases'] += len(cases['cases'])
    verify_scope(entries, metadata, manifest, scope, inventory)
    return counts
