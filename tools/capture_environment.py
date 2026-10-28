"""Capture the selected installed runtime; no installations and no NPU execution."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opbench.environment import audit, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'environment.local.json')
    args = parser.parse_args()
    if sys.platform != 'linux' or not sys.flags.isolated or site.ENABLE_USER_SITE:
        parser.error('Use the isolated runtime Python on Linux with -I')
    if sys.prefix in ('/usr', '/usr/local') or os.environ.get('CONDA_DEFAULT_ENV') == 'base':
        parser.error('Select an isolated runtime, not global Python or Conda base')
    if not os.environ.get('ASCEND_HOME_PATH'):
        parser.error('Select CANN with its absolute set_env.sh first')
    for key, value in os.environ.items():
        if value and (key in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD') or
                      key.startswith(('TRITON_','TORCHINDUCTOR_','RANK','WORLD_SIZE','LOCAL_RANK','MASTER_','HCCL_'))):
            parser.error('Remove inherited override before capture: ' + key)
    versions = {name:importlib.import_module(name).__version__ for name in ('torch','torch_npu','triton')}
    versions['triton-ascend'] = importlib.metadata.version('triton-ascend')
    from triton.backends.ascend import utils
    compiler, _ = utils._get_npucompiler_path()
    lib = importlib.import_module('triton._C.libtriton')
    result = dict(python_prefix=sys.prefix, versions=versions,
                  libtriton_sha256=digest(lib.__file__), compiler=compiler,
                  compiler_sha256=digest(compiler), cann_root=os.environ['ASCEND_HOME_PATH'],
                  environment={key:os.environ.get(key, '') for key in ('PATH','LD_LIBRARY_PATH')})
    audit(result)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
