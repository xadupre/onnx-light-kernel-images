# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Check release metadata and the matching onnx-light runtime dependency."""

import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from onnx_light_kernel_images import __version__

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPackageMetadata(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(__version__, "0.2.1")
        for filename, declaration in (
            ("pyproject.toml", f'version = "{__version__}"'),
            ("setup.py", f'version="{__version__}"'),
            (
                "CMakeLists.txt",
                f"project(onnx_light_kernel_images VERSION {__version__} LANGUAGES CXX)",
            ),
        ):
            with self.subTest(filename=filename):
                self.assertIn(declaration, (_REPO_ROOT / filename).read_text())

    def test_onnx_light_dependency(self):
        metadata = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
        self.assertIn("onnx-light==0.2.0", metadata["project"]["dependencies"])
        self.assertIn("onnx-light==0.2.0", metadata["build-system"]["requires"])
        for filename in ("ci_core.yml", "docs.yml"):
            with self.subTest(workflow=filename):
                self.assertIn(
                    'pip install "onnx-light==0.2.0"',
                    (_REPO_ROOT / ".github" / "workflows" / filename).read_text(),
                )
        self.assertIn(
            "Install onnx-light==0.2.0 before building the Python extension.",
            (_REPO_ROOT / "cmake" / "OnnxLightPython.cmake").read_text(),
        )

    def test_release_wheel_discovery(self):
        for filename in ("ci_core.yml", "docs.yml", "build_release_wheel.yml"):
            with self.subTest(workflow=filename):
                self.assertIn(
                    "  PIP_FIND_LINKS: "
                    "https://github.com/xadupre/onnx-light/releases/expanded_assets/0.2.0",
                    (_REPO_ROOT / ".github" / "workflows" / filename).read_text(),
                )

    def test_cpp_release_archive(self):
        self.assertIn(
            "https://github.com/xadupre/onnx-light/releases/download/0.2.0/"
            "${_onnx_light_archive}",
            (_REPO_ROOT / "CMakeLists.txt").read_text(),
        )

    def test_supported_python_versions(self):
        project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(project["requires-python"], ">=3.12")
        self.assertIn(
            'python-version: ["3.12", "3.13"]',
            (_REPO_ROOT / ".github" / "workflows" / "build_release_wheel.yml").read_text(),
        )

    def test_python_binding_abi_requires_explicit_metadata_or_override(self):
        abi_module = _REPO_ROOT / "cmake" / "OnnxLightPythonABI.cmake"
        cases = (
            ({}, "", None),
            ({"python_stable_abi": "unknown"}, "", None),
            ({"python_stable_abi": "ON"}, "", "ON"),
            ({"python_stable_abi": "OFF"}, "", "OFF"),
            ({"python_stable_abi": " on "}, "", "ON"),
            ({}, "ON", "ON"),
            ({}, "OFF", "OFF"),
        )
        for metadata, override, expected in cases:
            with (
                self.subTest(metadata=metadata, override=override),
                tempfile.TemporaryDirectory() as d,
            ):
                root = Path(d)
                (root / "onnx_light.py").write_text(
                    f"def get_cpp_build_info():\n    return {metadata!r}\n",
                    encoding="utf-8",
                )
                script = root / "probe.cmake"
                script.write_text(
                    f'set(Python_EXECUTABLE "{Path(sys.executable).as_posix()}")\n'
                    f'set(ONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI "{override}")\n'
                    f'include("{abi_module.as_posix()}")\n'
                    'message(STATUS "Detected ABI: ${_onnx_light_stable_abi}")\n',
                    encoding="utf-8",
                )
                result = subprocess.run(
                    ["cmake", "-P", str(script)],
                    cwd=root,
                    env={**os.environ, "PYTHONPATH": str(root)},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if expected is None:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "Set ONNX_LIGHT_KERNEL_IMAGES_PYTHON_STABLE_ABI to ON or OFF.",
                        " ".join(result.stderr.split()),
                    )
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"Detected ABI: {expected}", result.stdout)


if __name__ == "__main__":
    unittest.main()
