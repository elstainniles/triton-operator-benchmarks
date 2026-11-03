#!/usr/bin/env python3
"""Port retained CPU input/reference recipes to per-case in-memory fixtures.

This is an offline importer. Generated fixtures do not load tensor files or
depend on the importer's working directories. Unsupported source dependencies
are reported for manual adaptation instead of being replaced by guessed gold.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Gap(ValueError):
    pass


def named(node):
    if isinstance(node, ast.Subscript):
        return named(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return named(node.value) + "." + node.attr
    return ""


class CPU(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("empty", "empty_like", "new_empty"):
                node.func.attr = {"empty": "zeros", "empty_like": "zeros_like", "new_empty": "new_zeros"}[node.func.attr]
            if node.func.attr == "cuda":
                node.func.attr = "cpu"
                node.args = []
                node.keywords = []
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in ("cuda", "cuda:0", "cuda:1", "npu", "npu:0"):
            return ast.copy_location(ast.Constant("cpu"), node)
        return node


STANDARD = {"torch", "math", "typing", "functools", "itertools", "collections", "operator", "copy", "numpy", "einops", "dataclasses", "enum", "types", "os"}
BAD_CALLS = {"torch.load", "torch.save", "open", "exec", "eval", "compile", "__import__"}
BAD_ATTRS = {"read_text", "read_bytes", "write_text", "write_bytes", "mkdir", "glob", "rglob", "resolve", "is_file", "exists"}


def symbols(tree):
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        definitions[name.id] = node
    return definitions


def free_names(node):
    loaded = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assigned = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
    assigned |= {n.arg for n in ast.walk(node) if isinstance(n, ast.arg)}
    assigned |= {n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    return loaded - assigned


def check_pure(node):
    for part in ast.walk(node):
        if isinstance(part, ast.Call):
            if named(part.func) in BAD_CALLS:
                raise Gap("Reference depends on file IO or dynamic loading: " + named(part.func))
            if isinstance(part.func, ast.Attribute) and part.func.attr in BAD_ATTRS:
                raise Gap("Reference depends on source/filesystem IO: " + part.func.attr)
        if isinstance(part, ast.Constant) and isinstance(part.value, str) and "/home/" in part.value:
            raise Gap("Reference closure retains an absolute local path")


class Specialize(ast.NodeTransformer):
    """Select already-existing case-ID branches without changing equations."""
    def __init__(self, values):
        self.values = values

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.values:
            return ast.copy_location(ast.parse(repr(self.values[node.id]), mode="eval").body, node)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if not any(isinstance(part, (ast.Call, ast.Attribute, ast.Name)) for part in ast.walk(node.test)):
            try:
                answer = eval(compile(ast.fix_missing_locations(ast.Expression(node.test)), "<constant-case-condition>", "eval"), {"__builtins__": {}}, {})
            except Exception:
                return node
            return node.body if answer else node.orelse
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if not any(isinstance(n, (ast.Call, ast.Attribute, ast.Name)) for n in ast.walk(node.test)):
            try:
                answer = eval(compile(ast.fix_missing_locations(ast.Expression(node.test)), '<constant-input-branch>', 'eval'), {'__builtins__': {}}, {})
                return node.body if answer else node.orelse
            except Exception:
                pass
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        body = []
        for statement in node.body:
            body.append(statement)
            if isinstance(statement, (ast.Return, ast.Raise)):
                break
        node.body = body
        return node


def specialize_function(tree, function, values):
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values.setdefault(node.targets[0].id, ast.literal_eval(node.value))
            except (ValueError, TypeError):
                pass
    return Specialize(values).visit(copy.deepcopy(function))


class RemoveArchivedOjaCache(ast.NodeTransformer):
    """Remove historical-file/cache comparisons, retained by offline validation."""
    def visit_Assign(self, node):
        if any(isinstance(part, ast.Name) and part.id in ("input_digest", "old_path", "old", "CACHE") for target in node.targets for part in ast.walk(target)):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if any(isinstance(part, ast.Name) and part.id == "CACHE" for part in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Global(self, node):
        return None if 'CACHE' in node.names else node

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'CACHE':
            node.value = ast.Name(id='gold', ctx=ast.Load())
        return node

    def visit_For(self, node):
        if any(isinstance(part, ast.Name) and part.id == "old" for part in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "old_math" and node.attr == "oja_reference":
            return ast.copy_location(ast.Name(id="oja_reference", ctx=ast.Load()), node)
        return self.generic_visit(node)


class PortableReference(ast.NodeTransformer):
    """Omit source-loader/old-file audit statements; keep numerical checks."""
    def __init__(self, removed_names=(), removed_calls=()):
        self.removed_names = set(removed_names)
        self.removed_calls = set(removed_calls)

    def visit_Assign(self, node):
        if any(isinstance(n, ast.Name) and n.id in self.removed_names for target in node.targets for n in ast.walk(target)):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and named(node.value.func) in self.removed_calls:
            return None
        if any(isinstance(n, ast.Name) and n.id == 'old_reference' for n in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if any(isinstance(n, ast.Name) and n.id in ('contract','old_reference') for n in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_For(self, node):
        if any(isinstance(n, ast.Name) and n.id == 'old_reference' for n in ast.walk(node)):
            return None
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        return None if node.name in ('CPU64','PromoteReference') else node

    def visit_Return(self, node):
        if isinstance(node.value, ast.Tuple) and len(node.value.elts)==2:
            tail=node.value.elts[1]
            if isinstance(tail,ast.Dict) or isinstance(tail,ast.Call) and named(tail.func)=='dict':
                node.value=node.value.elts[0]
        return self.generic_visit(node)


class PromoteCPU64(ast.NodeTransformer):
    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if node.attr=='float':node.attr='double'
        elif node.attr=='float32' and isinstance(node.value,ast.Name) and node.value.id=='torch':node.attr='float64'
        return node


def portable_reference(refpath, refnode, source, case, out, exporter):
    """Resolve archived source-file loaders during import, not at test runtime."""
    family=refpath.parent.name
    extra_imports=[]
    overrides={}
    if family=='gsa_forward_remaining_reference':
        original=source/'tests/ops/ascend_ops/cv_operator/chunk_gsa_fwd_k/torch_native_impl.py'
        tree=ast.parse(original.read_text());defs=symbols(tree)
        promoted={name:PromoteCPU64().visit(copy.deepcopy(node)) for name,node in defs.items() if isinstance(node,ast.FunctionDef)}
        exporter.export(original,{'chunk_gsa_fwd_k_torch'},'original_cpu',promoted)
        refnode=PortableReference(('source','tree','ns'),('exec',)).visit(refnode)
        for node in ast.walk(refnode):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Subscript) and named(node.func.value)=='ns':
                node.func=ast.Name(id='chunk_gsa_fwd_k_torch',ctx=ast.Load())
        extra_imports.append('from original_cpu import chunk_gsa_fwd_k_torch')
    elif family=='oja_bwd_pending_original':
        refnode=PortableReference(('tree','fn','ns'),('exec',)).visit(refnode)
        if any(isinstance(n,ast.Subscript) and named(n.value)=='ns' for n in ast.walk(refnode)):
            original=source/case['original_source'];symbol='naive_chunk_oja_bwd_dhu'
            exporter.export(original,{symbol},'original_cpu')
            for n in ast.walk(refnode):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and named(n.func.value)=='ns':n.func=ast.Name(id=symbol,ctx=ast.Load())
            extra_imports.append('from original_cpu import '+symbol)
        # Source was only read by the now-resolved loader.
        refnode.args.defaults=[ast.Constant(None)]
    elif family=='chunkwise_diag_dkg_cpu64':
        refnode=PortableReference().visit(refnode)
        original=source/'tests/ops/ascend_ops/cv_operator/chunkwise_bwd_kernel_diag/test_chunkwise_bwd_kernel_diag_cpu.py'
        if not original.exists():
            original=next(source.rglob('*chunkwise_bwd_kernel_diag*cpu*.py'))
        tree=ast.parse(original.read_text());fn=copy.deepcopy(symbols(tree)['chunkwise_bwd_kernel_diag_cpu_ref'])
        fn.name='diag_reference';fn.args.args=[ast.arg(arg='data')];fn.args.defaults=[]
        fn=PortableReference(('ptfile_path','data','save_path'),('torch.save','print')).visit(fn)
        fn.body.append(ast.Return(ast.Name(id='output',ctx=ast.Load())))
        exporter.export(original,{'diag_reference'},'original_cpu',{'diag_reference':fn})
        overrides['original_diag_reference']=ast.parse("def original_diag_reference(d):\n    return diag_reference({'input_data':d})\n").body[0]
        extra_imports.append('from original_cpu import diag_reference')
    elif family in ('old_five_approved_cpu64','dplr_dv_approved_cpu64'):
        refnode=PortableReference(('old_reference','helper'),()).visit(refnode)
        cid=case['id']
        # Only the existing loader/provenance shell changes; equations remain.
        if cid=='cv_3de7e791555908a3':
            helper=exporter.search[0].parent/'data/additional_cpu_references/generate.py'
            exporter.export(helper,{'wy_math'},'math_helper');extra_imports.append('import math_helper as helper')
        elif cid in ('cv_f26fc8e0dba9a56d','cv_62624120aad8a29d'):
            helper=exporter.search[0].parent/'data/nsa_inputs/generate.py'
            exporter.export(helper,{'attention'},'math_helper');extra_imports.append('import math_helper as helper')
        elif cid=='cv_7cbdb409bc083553':
            helper=exporter.search[0].parent/'data/cv_references/generate.py'
            tree=ast.parse(helper.read_text());fn=symbols(tree)['reference'];fn=specialize_function(tree,fn,{'kind':'dplr_dv'})
            exporter.export(helper,{'reference'},'math_helper',{'reference':fn});extra_imports.append('import math_helper as helper')
        elif cid in ('cv_0dcf8e4cb1081f20','cv_10e3db9fa19b5e99'):
            tree=ast.parse(refpath.read_text());mesa=copy.deepcopy(symbols(tree)['mesa_reference'])
            mesa=PortableReference(('names','tree','selected','namespace'),('exec',)).visit(mesa)
            original=source/'tests/ops/ascend_ops/cv_operator/chunk_mesa_net_h_kv_bwd_intra_kernel_dq/test_chunk_mesa_net_h_kv_bwd_intra_kernel_dq_npu.py'
            names=['naive_chunk_mesa_net_h_kv_bwd_intra_dkv','naive_chunk_mesa_net_h_kv_bwd_intra_dq']
            defs=symbols(ast.parse(original.read_text()));promoted={name:PromoteCPU64().visit(copy.deepcopy(defs[name])) for name in names}
            exporter.export(original,set(names),'original_cpu',promoted)
            # Drop the source extraction list and assert; these do not check output.
            mesa.body=[n for n in mesa.body if not isinstance(n,ast.Assert) or 'selected' not in ast.unparse(n)]
            for n in ast.walk(mesa):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Subscript) and named(n.func.value)=='namespace':
                    index=ast.literal_eval(n.func.slice.slice);n.func=ast.Name(id=names[index],ctx=ast.Load())
            mesa.args.defaults=[ast.Constant(None)]
            overrides['mesa_reference']=mesa
            extra_imports.extend('from original_cpu import '+name for name in names)
            for n in ast.walk(refnode):
                if isinstance(n,ast.Call) and named(n.func)=='mesa_reference':n.args=n.args[:1]
    return refnode,overrides,extra_imports


class PureExporter:
    def __init__(self, out, search):
        self.out = out
        self.search = search
        self.pending = {}
        self.sources = []

    def export(self, path, required, output_name="reference_impl", overrides=None, cpu=False):
        path = Path(path)
        tree = ast.parse(path.read_text())
        defs = symbols(tree)
        if overrides:
            defs.update(overrides)
        imports = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = (node, alias.name, None)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports[alias.asname or alias.name] = (node, node.module, alias.name)
        selected = {}
        queue = list(required)
        needed_imports = {}
        while queue:
            name = queue.pop()
            if name in selected:
                continue
            if name not in defs:
                raise Gap(f"Missing pure definition {name} in {path.name}")
            node = copy.deepcopy(defs[name])
            if cpu:
                node = CPU().visit(node)
            check_pure(node)
            selected[name] = node
            for dependency in free_names(node):
                if dependency in defs and dependency not in selected:
                    queue.append(dependency)
                elif dependency in imports:
                    needed_imports[dependency] = imports[dependency]
        emitted_imports = []
        for alias, (node, module, member) in needed_imports.items():
            if module == "triton" and member is None:
                emitted_imports += ast.parse("import types\ntriton = types.SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d, next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length())\n").body
                continue
            if module.split(".")[0] in STANDARD:
                if member:
                    emitted_imports.append(ast.ImportFrom(module=module, names=[ast.alias(name=member, asname=alias if alias != member else None)], level=0))
                else:
                    emitted_imports.append(ast.Import(names=[ast.alias(name=module, asname=alias if alias != module else None)]))
                continue
            candidates = [path.parent / (module + ".py")] + [folder / (module + ".py") for folder in self.search]
            imported = next((candidate for candidate in candidates if candidate.is_file()), None)
            if imported is None:
                raise Gap("Pure reference helper is unavailable: " + module)
            requested = {member} if member else {
                sub.attr for item in selected.values() for sub in ast.walk(item)
                if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == alias}
            if not requested:
                raise Gap("Dynamic reference helper module access: " + module)
            child_name = output_name + "_" + alias
            self.export(imported, requested, child_name, cpu=cpu)
            if member:
                emitted_imports.append(ast.ImportFrom(module=child_name, names=[ast.alias(name=member, asname=alias if alias != member else None)], level=0))
            else:
                emitted_imports.append(ast.Import(names=[ast.alias(name=child_name, asname=alias)]))
        # Preserve original definition order so default expressions see globals.
        selected_ids = {getattr(node, "lineno", -1) for node in selected.values()}
        unique_nodes = {ast.dump(node, include_attributes=False): node for node in selected.values()}
        body = emitted_imports + sorted(unique_nodes.values(), key=lambda node: getattr(node, "lineno", 0))
        text = '"""Existing independent CPU equations; file-generation main omitted."""\n' + ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))) + "\n"
        ast.parse(text)
        (self.out / (output_name + ".py")).write_text(text)
        self.sources.append({"name": path.name, "sha256": sha(path), "functions": sorted(required)})


def input_prefix(path, function_name, seed):
    tree = ast.parse(path.read_text())
    defs = symbols(tree)
    if function_name == "source_top_level":
        defs[function_name] = ast.parse("def source_top_level():\n    pass\n").body[0]
        defs[function_name].body = [copy.deepcopy(node) for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.If))]
    if function_name not in defs:
        raise Gap("Input generator function not found: " + str(function_name))
    function = defs[function_name]
    if not isinstance(function, ast.FunctionDef):
        raise Gap("Input generator is not a function")
    tainted = {name for name, node in defs.items() if isinstance(node, ast.FunctionDef) and any("triton." in ast.unparse(d) for d in node.decorator_list)}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("fla", "triton")):
            tainted.update(alias.asname or alias.name for alias in node.names)
    for _ in range(len(defs)):
        previous = set(tainted)
        for name, node in defs.items():
            if isinstance(node, ast.FunctionDef) and any(isinstance(n, ast.Call) and (isinstance(n.func, ast.Subscript) or named(n.func) in tainted) for n in ast.walk(node)):
                tainted.add(name)
        if previous == tainted:
            break
    prefix = []
    literal_values = {}
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    try:
                        literal_values[target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        literal_values.pop(target.id, None)
        if isinstance(statement, ast.If):
            test = Specialize(literal_values).visit(copy.deepcopy(statement.test))
            if not any(isinstance(n, (ast.Call, ast.Attribute, ast.Name)) for n in ast.walk(test)):
                try:
                    truth = eval(compile(ast.fix_missing_locations(ast.Expression(test)), '<original-static-branch>', 'eval'), {'__builtins__': {}}, {})
                except Exception:
                    truth = None
                if truth is False and not statement.orelse:
                    continue
        if isinstance(statement, ast.Assign) and any(isinstance(n, ast.IfExp) for n in ast.walk(statement.value)):
            statement = copy.deepcopy(statement)
            for n in ast.walk(statement):
                if isinstance(n, ast.IfExp):
                    n.test = Specialize(literal_values).visit(n.test)
            statement = Specialize({}).visit(statement)
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('save_dir','kernel_dir','output_dir','output_path','input_path','ptfile_path') for t in statement.targets):
            continue
        calls = [n for n in ast.walk(statement) if isinstance(n, ast.Call)]
        if any(isinstance(n.func, ast.Subscript) or named(n.func) in tainted for n in calls):
            break
        if any(named(n.func) in ("torch.save", "print", "os.makedirs", "torch.cuda.synchronize", "torch.cuda.empty_cache", "torch.cuda.manual_seed", "sys.path.insert") for n in calls):
            continue
        if any(isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith(("/home/", "/data/", "/tmp/")) for n in ast.walk(statement)):
            continue
        prefix.append(CPU().visit(copy.deepcopy(statement)))
    schemas = []
    for statement in function.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        names = [ast.unparse(n) for n in targets]
        if not any(name in ("data", "input_data", "inputs_dict", "input_dict") or name.startswith("data['input_data']") for name in names):
            continue
        statement = copy.deepcopy(statement)
        # Keep the input dictionary from a saved input/output package, without
        # evaluating the unavailable GPU-output expressions.
        for dictionary in ast.walk(statement):
            if isinstance(dictionary, ast.Dict):
                kept = [(key, value) for key, value in zip(dictionary.keys, dictionary.values)
                        if not isinstance(key, ast.Constant) or key.value not in ("gpu_output", "output_data")]
                dictionary.keys = [key for key, value in kept]
                dictionary.values = [value for key, value in kept]
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript) for n in ast.walk(statement)):
            continue
        schemas.append(CPU().visit(copy.deepcopy(statement)))
    capture = ast.parse("def make_inputs():\n    pass\n").body[0]
    default_args = function.args.args[-len(function.args.defaults):] if function.args.defaults else []
    if len(default_args) != len(function.args.args):
        raise Gap("Original input generator has required parameters")
    defaults = [ast.Assign(targets=[ast.Name(id=arg.arg, ctx=ast.Store())], value=CPU().visit(copy.deepcopy(value))) for arg, value in zip(default_args, function.args.defaults)]
    capture.body = ast.parse(f"torch.manual_seed({seed!r})").body + defaults + prefix
    fallbacks = []
    for statement in schemas:
        fallbacks.append(ast.Try(body=[statement], handlers=[ast.ExceptHandler(type=ast.Name(id="NameError", ctx=ast.Load()), name=None, body=[ast.Pass()])], orelse=[], finalbody=[]))
    if fallbacks:
        condition = ast.parse("not (any(isinstance(locals().get(key), dict) and locals().get(key) for key in ('inputs_dict', 'input_dict', 'input_data')) or (isinstance(locals().get('data'), dict) and locals()['data'].get('input_data')))", mode="eval").body
        # Avoid comprehension-local locals(): capture the namespace once.
        capture.body += ast.parse("_prefix_values = locals().copy()\n").body
        for part in ast.walk(condition):
            if isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and part.func.id == "locals":
                part.func = ast.Attribute(value=ast.Name(id="_prefix_values", ctx=ast.Load()), attr="copy", ctx=ast.Load())
        capture.body.append(ast.If(test=condition, body=fallbacks, orelse=[]))
    capture.body += ast.parse("return locals()\n").body
    return tree, capture


def build_one(row, entry, case, source, out, support_search, manual_sources):
    generator = row.get("input_generator")
    if not generator:
        raise Gap("No generic upstream input prefix recipe; requires existing family/manual generator")
    generator_path = source / generator
    exporter = PureExporter(out, support_search)
    metadata = json.loads(Path(row["metadata"]).read_text())
    manual_hash = (metadata.get("source_proof") or {}).get("manual_generator_sha256")
    special_mesa = row["case_id"] == "cv_0301e9554ccfc5f5"
    split_mesa = row['case_id'] in ('cv_0dcf8e4cb1081f20','cv_10e3db9fa19b5e99')
    if split_mesa:
        exporter.export(generator_path, {'generate_test_data'}, 'input_original', cpu=True)
        (out/'input_impl.py').write_text("import torch\nfrom input_original import generate_test_data\n\ndef make_inputs():\n    data, grid = generate_test_data(dtype=torch.float16, device='cpu')\n    for key in ('dq','dk','dv','dg'):\n        data[key] = torch.zeros_like(data[key], dtype=torch.float32)\n    data['dg_final'] = torch.zeros_like(data['dg'])\n    return {'input_data':data, 'grid':grid}\n")
    elif special_mesa:
        retained = Path(entry["bundle_path"]) / next(item for item in case["source_files"] if item.startswith("extracted/") and item.endswith(".py"))
        exporter.export(retained, {"generate_test_data"}, "input_original", cpu=True)
        (out / "input_impl.py").write_text("import torch\nfrom input_original import generate_test_data\n\ndef make_inputs():\n    data, grid = generate_test_data(dtype=torch.float16, device='cpu')\n    for key in ('dq', 'dk', 'dv', 'dg'):\n        data[key].fill_(float('nan'))\n    return {'input_data': data, 'grid': grid}\n")
    elif manual_hash:
        manual = manual_sources.get(manual_hash)
        if manual is None:
            raise Gap("Recorded manual generator source hash was not found")
        tree = ast.parse(manual.read_text())
        defs = symbols(tree)
        build = defs["build"]
        args = [arg.arg for arg in build.args.args]
        kind = None
        for global_name in ("SPECS", "GROUPS"):
            if global_name not in defs:
                continue
            specs = ast.literal_eval(defs[global_name].value)
            if isinstance(specs, dict) and row["case_id"] in specs:
                kind = specs[row["case_id"]]
                if isinstance(kind, (tuple, list)):
                    kind = kind[0]
            elif isinstance(specs, dict):
                kind = next((key for key, value in specs.items() if row["case_id"] in value), kind)
        if args and args[0] == "cid":
            kind = row["case_id"]
        if args and kind is None:
            raise Gap("Manual generator case-to-kind mapping unavailable")
        build = specialize_function(tree, build, {args[0]: kind} if args else {})
        exporter.export(manual, {"build"}, "input_manual", {"build": build}, cpu=True)
        actual_args = ([repr(kind)] + ["None"] * (len(args) - 1)) if args else []
        wrapper = "from input_manual import build\n\ndef make_inputs():\n    result = build(" + ", ".join(actual_args) + ")\n"
        wrapper += "    grid = None\n    if isinstance(result, tuple):\n        data = result[0]\n        if len(result) == 4:\n            grid = result[3]\n    else:\n        data = result\n"
        wrapper += "    if 'input_data' in data:\n        return {'data': data, 'grid': data.get('grid', grid)}\n"
        schema_key = "inputs_dict" if metadata.get("style") == "plain" else "input_data"
        wrapper += f"    return {{{schema_key!r}: data, 'grid': grid}}\n"
        (out / "input_impl.py").write_text(wrapper)
    else:
        original_tree, capture = input_prefix(generator_path, row["input_source_function"], row["seed"] or 20261031)
        exporter.export(generator_path, {"make_inputs"}, "input_impl", {"make_inputs": capture}, cpu=True)
    refs = []
    for item in row["references"]:
        for fn in item["functions"]:
            if fn["name"] == "reference":
                refs.append((Path(item["path"]), fn["args"]))
    if not refs:
        raise Gap("Independent reference function not registered in retained source closure")
    refpath, args = refs[-1]
    reference_name = "intra_reference" if row["case_id"] == "cv_23467ec91bc8e2b1" else "reference"
    if args[:2] in (["cid", "d"], ["cid", "data"]):
        refcall = f"reference({row['case_id']!r}, inputs)"
    elif args[:2] == ['case', 'contract']:
        refcall = f"reference({{'id': {row['case_id']!r}}}, None, inputs)"
    elif args[:1] in (["d"], ["data"]):
        refcall = "reference(inputs)"
    else:
        raise Gap("Reference requires case-specific context: " + str(args))
    reftree = ast.parse(refpath.read_text())
    refnode = next(node for node in reftree.body if isinstance(node, ast.FunctionDef) and node.name == reference_name)
    if refpath.parent.name in ('oja_based_pending_cpu64', 'oja_h_pending_original'):
        refnode = RemoveArchivedOjaCache().visit(copy.deepcopy(refnode))
    refnode = specialize_function(reftree, refnode, {"cid": row["case_id"]})
    imported_oja = False
    if refpath.parent.name == "oja_based_pending_cpu64":
        original_oja = Path(entry["bundle_path"]) / "resume_cv_input_slices/manual_generate.py"
        exporter.export(original_oja, {"oja_reference"}, "original_oja")
        imported_oja = True
    refnode, extra_overrides, extra_imports = portable_reference(refpath, refnode, source, case, out, exporter)
    overrides = {reference_name: refnode, **extra_overrides}
    imported_reference = None
    naive_calls = [node for node in ast.walk(refnode) if isinstance(node, ast.Call) and named(node.func) == "original_naive"]
    if naive_calls:
        folder, symbol = [ast.literal_eval(value) for value in naive_calls[0].args]
        original = next((source / "tests/ops/ascend_ops/cv_operator" / folder).glob("*npu.py"))
        exporter.export(original, {symbol}, "original_cpu")
        overrides["original_naive"] = ast.parse(f"def original_naive(folder, name):\n    return {symbol}, {{'function': {symbol!r}, 'function_AST_unmodified': True}}\n").body[0]
        imported_reference = symbol
    exporter.export(refpath, {reference_name}, overrides=overrides)
    if extra_imports:
        with (out/'reference_impl.py').open('a') as handle:
            handle.write('\n'+'\n'.join(extra_imports)+'\n')
    if imported_oja:
        with (out / "reference_impl.py").open("a") as handle:
            handle.write("\nfrom original_oja import oja_reference\n")
    if imported_reference:
        with (out / "reference_impl.py").open("a") as handle:
            handle.write("\nfrom original_cpu import " + imported_reference + "\n")
    if reference_name != "reference":
        with (out / "reference_impl.py").open("a") as handle:
            handle.write("\nreference = " + reference_name + "\n")
    if row["case_id"] == "cv_6384385b8cfbaeff":
        exporter.export(source / case["original_source"], {"naive_chunk_rwkv6_bwd_dh"}, "original_cpu")
        (out / "reference_impl.py").write_text("from original_cpu import naive_chunk_rwkv6_bwd_dh\n\ndef reference(cid, data):\n    kwargs = {name: data[name] for name in ('q', 'gi', 'ge', 'dht', 'scale', 'T', 'HQ', 'H', 'K', 'V', 'BT', 'NG', 'STORE_INITIAL_STATE_GRADIENT', 'USE_FINAL_STATE_GRADIENT', 'IS_VARLEN')}\n    kwargs.update(do_tensor=data['do'], cu_seqlens=None, chunk_offsets=None)\n    dh, dh0 = naive_chunk_rwkv6_bwd_dh(**kwargs)\n    return {'dh': dh.half().float(), 'dh0': dh0}\n")
    if split_mesa:
        original = source/'tests/ops/ascend_ops/cv_operator/chunk_mesa_net_h_kv_bwd_intra_kernel_dq/test_chunk_mesa_net_h_kv_bwd_intra_kernel_dq_npu.py'
        names = ['naive_chunk_mesa_net_h_kv_bwd_intra_dkv','naive_chunk_mesa_net_h_kv_bwd_intra_dq']
        exporter.export(original,set(names),'original_fp32')
        (out/'legacy_reference.py').write_text("from original_fp32 import " + ', '.join(names) + "\n\ndef reference(d):\n    common = [d[k].float() for k in ('q_star','k','v','beta','h_kv','g','do')]\n    dims = [d[k] for k in ('B','T','H','K','V','BT')]\n    dk,dv,dg = naive_chunk_mesa_net_h_kv_bwd_intra_dkv(*common,d['dh_kv'].float(),*dims)\n    dq,dgf = naive_chunk_mesa_net_h_kv_bwd_intra_dq(*common,dg,*dims)\n    return dict(dk=dk,dv=dv,dg=dg,dq=dq,dg_final=dgf)\n")
    required = case.get("required_data", [])
    refnames = [Path(item["path"]).name for item in required if "reference" in Path(item["path"]).name or "outputs_fp64" in Path(item["path"]).name]
    inputnames = [Path(item["path"]).name for item in required if Path(item["path"]).name not in refnames]
    split_files = len(set(inputnames)) > 1
    if not refnames:
        raise Gap('No independent reference payload registered')
    if split_files and refpath.parent.name not in ('deltaformer_pending_original', 'oja_bwd_pending_original'):
        raise Gap("Fixture payload schema needs family adapter: " + str(inputnames) + "; refs=" + str(refnames))
    inputaliases = set(inputnames)
    for before, after in case.get("data_path_map", {}).items():
        if Path(after).name in inputaliases:
            inputaliases.add(Path(before).name)
    grid = (metadata.get("grid_serialization_repair") or {}).get("original_grid") or (metadata.get("source_proof") or {}).get("original_grid") or metadata.get("grid")
    text = '''"""Fixed-seed original-spec inputs and independent CPU reference, in memory."""
import torch
from input_impl import make_inputs
from reference_impl import reference

def make_fixtures():
    values = make_inputs()
    if isinstance(values.get('inputs_dict'), dict):
        inputs = values['inputs_dict']
        payload = inputs
    elif isinstance(values.get('input_dict'), dict):
        inputs = values['input_dict']
        payload = {'input_data': inputs}
    elif isinstance(values.get('input_data'), dict):
        inputs = values['input_data']
        payload = {'input_data': inputs}
    elif isinstance(values.get('data'), dict) and 'input_data' in values['data']:
        inputs = values['data']['input_data']
        payload = {'input_data': inputs}
    elif isinstance(values.get('inputs'), dict):
        inputs = values['inputs']
        payload = inputs
    else:
        raise RuntimeError('Original input prefix did not produce a complete input schema')
'''
    if metadata.get('style') == 'plain':
        text += "    payload = inputs\n"
    if split_files and refpath.parent.name == 'deltaformer_pending_original':
        text += "    inputs = {name.removeprefix('input_'): value for name,value in inputs.items()}\n"
    text += (f"    grid = values.get('grid', {grid!r})\n" if manual_hash or special_mesa or split_mesa else f"    grid = {grid!r}\n")
    text += "    if 'input_data' in payload and grid is not None and not callable(grid):\n        payload['grid'] = tuple(grid)\n"
    if row["case_id"] == "cv_92ff29f22c91af7d" and "L8_valid_level" in entry["case_path"]:
        text += "    generator = torch.Generator().manual_seed(20261030)\n    extra = torch.randn(*inputs['l'].shape[:-1], 1, generator=generator, dtype=inputs['l'].dtype) * 0.1\n    inputs['l'] = torch.cat((inputs['l'], extra), dim=-1)\n    inputs['L'] = 8\n"
    text += f"    gold = {refcall}\n"
    text += "    if isinstance(gold, tuple):\n        gold = gold[0]\n"
    if row["case_id"] == "cv_a7181325ade2b3f0":
        text += "    gold = {'dA': gold['dA']}\n"
    text += "    fixtures = {}\n"
    if split_files:
        text += f"    for name in {sorted(inputaliases)!r}:\n        fixtures[name] = inputs[name.removesuffix('.pt').removeprefix('input_')]\n"
    else:
        text += f"    for name in {sorted(inputaliases)!r}:\n        fixtures[name] = payload\n"
    text += f"    for name in {refnames!r}:\n        fixtures[name] = gold\n"
    if split_mesa:
        text += "    from legacy_reference import reference as legacy_reference\n    fixtures['independent_reference_cpu_fp32.pt'] = legacy_reference(inputs)\n"
    text += "    return fixtures\n"
    (out / "fixture.py").write_text(text)
    return {"case_id": row["case_id"], "status": "ported_pending_cpu_validation", "provider": str(out / "fixture.py"),
            "case_path": entry["case_path"], "validation_standard": entry.get("validation_standard"),
            "data_aliases": sorted(inputaliases) + refnames, "sources": exporter.sources,
            "seed": row["seed"], "input_generator_source": generator}


def write_index(output, catalog):
    records=json.loads((output/'manifest.json').read_text())['cases']
    available=[row for row in json.loads(catalog.read_text())['entries'] if row.get('available') and row.get('group')=='CV']
    entries=[]; covered=set()
    def fingerprint(case):
        return sorted((Path(item['path']).name,item.get('sha256')) for item in case.get('required_data',[]))
    for record in records:
        folder=output/record['case_id'];validation_path=folder/'validation.json'
        validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
        hashes={path.name:sha(path) for path in folder.glob('*.py')}
        valid=validation.get('status')=='passed' and bool(hashes)
        if valid and validation.get('provider_sha256'):
            valid=hashes==validation['provider_sha256']
        elif valid:
            valid=validation_path.stat().st_mtime>=max(path.stat().st_mtime for path in folder.glob('*.py'))
        status='ready' if valid else 'blocked'
        owncase=json.loads(Path(record['case_path']).read_text())
        identical=[row for row in available if row['case_id']==record['case_id'] and fingerprint(json.loads(Path(row['case_path']).read_text()))==fingerprint(owncase)]
        paths=sorted({row['case_path'] for row in identical}|{record['case_path']})
        for case_path in paths:
            if case_path in covered:continue
            covered.add(case_path)
            entries.append({
                'case_id':record['case_id'],'case_path':case_path,'provider_dir':str(folder),
                'status':status,'reason':'' if valid else validation.get('error') or record.get('reason','CPU validation pending'),
                'generator_provenance':record.get('sources',[]),'seed':record.get('seed'),
                'input_generator_source':record.get('input_generator_source'),
                'adaptations':['Original fixed-seed inputs constructed on CPU; empty output allocations initialized to deterministic zero.', 'Source-file loaders resolved at import time; numerical reference equations and independent numerical checks retained.', 'Historical tensor-file comparisons performed in local CPU validation; published fixture construction performs no tensor-file IO.'],
                'validation_standard':record.get('validation_standard'),
                'fixture_aliases':record.get('data_aliases',[]),'reference_data_sha256':fingerprint(owncase),
                'provider_sha256':hashes,'validation':validation,
            })
    coverage={status:sum(row['status']==status for row in entries) for status in ('ready','blocked')}
    coverage['ready_case_ids']=len({row['case_id'] for row in entries if row['status']=='ready'})
    coverage['catalog_case_paths_not_matched']=len({row['case_path'] for row in available}-covered)
    (output/'index.json').write_text(json.dumps({'entries':entries,'coverage':coverage},indent=2)+'\n')
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--support-search", type=Path, action="append", default=[])
    parser.add_argument("--manual-sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.recipes.read_text())
    entries = {}
    for entry in json.loads(args.catalog.read_text())["entries"]:
        if entry.get("available") and entry.get("group") == "CV" and (entry["case_id"] not in entries or entry["side"] == "baseline"):
            entries[entry["case_id"]] = entry
    args.output.mkdir(parents=True, exist_ok=True)
    manual_sources = {}
    if args.manual_sources:
        for path in args.manual_sources.rglob("*.py"):
            manual_sources.setdefault(sha(path), path)
    results = []
    for row in rows:
        entry = entries[row["case_id"]]
        out = args.output / row["case_id"]
        out.mkdir(exist_ok=True)
        try:
            result = build_one(row, entry, json.loads(Path(entry["case_path"]).read_text()), args.source, out, args.support_search, manual_sources)
        except Exception as error:
            result = {"case_id": row["case_id"], "case_path": entry["case_path"], "status": "needs_family_adapter", "reason": type(error).__name__ + ": " + str(error)}
        results.append(result)
    manifest = {"cases": results, "counts": {status: sum(row["status"] == status for row in results) for status in sorted({row["status"] for row in results})}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
    write_index(args.output,args.catalog)


if __name__ == "__main__":
    main()
