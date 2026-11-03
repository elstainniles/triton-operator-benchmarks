"""Runtime isolation accepts standard venv layout without global packages."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from opbench.environment import ROOT, audit_python_paths


class PythonPathIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / 'runtime'
        self.base = self.root / 'base'
        self.bin = self.prefix / 'bin'
        self.bin.mkdir(parents=True)
        (self.base / 'bin').mkdir(parents=True)
        self.base_python = self.base / 'bin' / 'python'
        self.base_python.touch()
        self.python = self.bin / 'python'
        self.python.symlink_to(self.base_python)
        self.stdlib = self.base / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}'
        self.stdlib.mkdir(parents=True)
        self.zip = self.stdlib.parent / f'python{sys.version_info.major}{sys.version_info.minor}.zip'
        self.packages = self.prefix / 'lib' / self.stdlib.name / 'site-packages'
        self.packages.mkdir(parents=True)
        self.paths = [str(self.zip), str(self.stdlib), str(self.stdlib / 'lib-dynload'),
                      str(self.packages), str(ROOT)]

    def context(self, *, paths=None, executable=None):
        stack = ExitStack()
        stack.enter_context(patch('opbench.environment.sys.executable', str(executable or self.python)))
        stack.enter_context(patch('opbench.environment.sys.base_prefix', str(self.base)))
        stack.enter_context(patch('opbench.environment.sys._base_executable', str(self.base_python), create=True))
        stack.enter_context(patch('opbench.environment.sys.path', self.paths if paths is None else paths))
        stack.enter_context(patch('opbench.environment.sysconfig.get_path', return_value=str(self.stdlib)))
        stack.enter_context(patch('opbench.environment.sysconfig.get_config_var', return_value=str(self.stdlib / 'lib-dynload')))
        return stack

    def test_standard_venv_symlink_and_base_stdlib_are_allowed(self):
        with self.context():
            audit_python_paths(self.prefix)

    def test_copied_venv_interpreter_is_allowed(self):
        self.python.unlink()
        self.python.touch()
        with self.context():
            audit_python_paths(self.prefix)

    def test_external_site_and_dist_packages_are_rejected(self):
        for suffix in ('site-packages', 'dist-packages'):
            with self.subTest(suffix=suffix), self.context(paths=self.paths + [str(self.stdlib / suffix)]):
                with self.assertRaisesRegex(RuntimeError, 'Unexpected Python search path'):
                    audit_python_paths(self.prefix)

    def test_arbitrary_directory_below_stdlib_is_not_an_allowed_root(self):
        with self.context(paths=self.paths + [str(self.stdlib / 'foreign_source')]):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected Python search path'):
                audit_python_paths(self.prefix)

    def test_symlinked_package_search_path_cannot_escape_runtime(self):
        foreign = self.root / 'foreign_packages'
        foreign.mkdir()
        link = self.prefix / 'extra_packages'
        link.symlink_to(foreign)
        with self.context(paths=self.paths + [str(link)]):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected Python search path'):
                audit_python_paths(self.prefix)

    def test_symlink_to_unrelated_interpreter_is_rejected(self):
        other = self.root / 'other_python'
        other.touch()
        self.python.unlink()
        self.python.symlink_to(other)
        with self.context():
            with self.assertRaisesRegex(RuntimeError, 'outside its base interpreter'):
                audit_python_paths(self.prefix)

    def test_invoking_base_interpreter_directly_is_rejected(self):
        with self.context(executable=self.base_python):
            with self.assertRaisesRegex(RuntimeError, 'executable outside runtime'):
                audit_python_paths(self.prefix)


if __name__ == '__main__':
    unittest.main()
