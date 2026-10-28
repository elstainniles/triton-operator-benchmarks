"""Audit existing isolated runtime. This module never builds or installs packages."""
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import site
import sys

from .catalog import ROOT


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def configure(expected, output, card):
    if sys.platform != 'linux' or not sys.flags.isolated or site.ENABLE_USER_SITE:
        raise RuntimeError('Run on Linux with the isolated runtime Python and -I')
    prefix = Path(expected['python_prefix']).resolve()
    if prefix in (Path('/usr'), Path('/usr/local'), Path('/')) or Path(sys.prefix).resolve() != prefix:
        raise RuntimeError('Runtime prefix mismatch or global Python')
    if not Path(sys.executable).resolve().is_relative_to(prefix):
        raise RuntimeError('Python executable outside runtime prefix')
    if os.environ.get('CONDA_DEFAULT_ENV') == 'base':
        raise RuntimeError('Conda base is not a test runtime')
    for key, value in os.environ.items():
        if value and (key in {'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'}
                      or key.startswith(('TRITON_', 'TORCHINDUCTOR_', 'RANK', 'WORLD_SIZE',
                                         'LOCAL_RANK', 'MASTER_', 'HCCL_'))):
            raise RuntimeError(f'Inherited runtime override must be removed: {key}')
    if any(not (Path(p).resolve().is_relative_to(prefix) or Path(p).resolve() == ROOT) for p in sys.path):
        raise RuntimeError('Unexpected Python search path outside runtime')
    cann = Path(expected['cann_root']).resolve()
    if not cann.is_dir() or Path(os.environ.get('ASCEND_HOME_PATH', '/missing')).resolve() != cann:
        raise RuntimeError('Explicit CANN selection missing or different')
    for key in ('PATH', 'LD_LIBRARY_PATH'):
        if os.environ.get(key, '') != expected['environment'][key]:
            raise RuntimeError(f'{key} differs from audited environment manifest')
    for key in ('libtriton_sha256', 'compiler_sha256'):
        if not re.fullmatch('[a-f0-9]{64}', expected[key]):
            raise ValueError('Expected a verified SHA256 for ' + key)
    cache = output / 'cache'
    cache.mkdir()
    for key, folder in [('TRITON_CACHE_DIR', 'triton'), ('TRITON_DUMP_DIR', 'dump'),
                        ('TRITON_OVERRIDE_DIR', 'override'), ('TMPDIR', 'tmp')]:
        path = cache / folder
        path.mkdir()
        os.environ[key] = str(path)
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(card)
    os.environ['ASCEND_DEVICE_ID'] = '0'
    os.environ['DEVICE_ID'] = '0'
    os.environ['OMP_NUM_THREADS'] = '4'


def audit(expected):
    prefix = Path(expected['python_prefix']).resolve()
    modules = {}
    for name in ('torch', 'torch_npu', 'triton', 'triton._C.libtriton'):
        mod = importlib.import_module(name)
        path = Path(mod.__file__).resolve()
        if not path.is_relative_to(prefix):
            raise RuntimeError(f'{name} imported outside runtime: {path}')
        modules[name] = dict(path=str(path), version=getattr(mod, '__version__', None))
    for name in ('torch', 'torch_npu', 'triton'):
        if modules[name]['version'] != expected['versions'][name]:
            raise RuntimeError('Package version mismatch: ' + name)
    ta = importlib.metadata.version('triton-ascend')
    if ta != expected['versions']['triton-ascend']:
        raise RuntimeError('TA package version mismatch')
    libhash = digest(modules['triton._C.libtriton']['path'])
    if libhash != expected['libtriton_sha256']:
        raise RuntimeError('libtriton binary hash mismatch')
    from triton.backends.ascend import utils
    compiler, _ = utils._get_npucompiler_path()
    if Path(compiler).resolve() != Path(expected['compiler']).resolve() or digest(compiler) != expected['compiler_sha256']:
        raise RuntimeError('Resolved Ascend compiler differs from manifest')
    linker = shutil.which('ld.lld')
    expected_linker = Path(expected['cann_root']) / 'tools/bisheng_compiler/bin/ld.lld'
    if not linker or Path(linker).resolve() != expected_linker.resolve():
        raise RuntimeError('Linker does not come from pinned CANN')
    return dict(python=sys.executable, prefix=sys.prefix, modules=modules, triton_ascend=ta,
                libtriton_sha256=libhash, compiler=compiler, compiler_sha256=digest(compiler),
                cann_root=expected['cann_root'], linker=linker,
                controlled_environment={k:os.environ.get(k) for k in
                    ('PATH', 'LD_LIBRARY_PATH', 'ASCEND_HOME_PATH', 'ASCEND_RT_VISIBLE_DEVICES',
                     'TRITON_CACHE_DIR', 'TMPDIR')})
