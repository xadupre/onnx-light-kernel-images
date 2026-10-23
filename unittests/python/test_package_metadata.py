# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Check release metadata and the matching onnx-light runtime dependency."""

import tomllib
import unittest
from pathlib import Path

from onnx_light_kernel_images import __version__

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPackageMetadata(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(__version__, "0.2.0")
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
        self.assertIn(
            'pip install "onnx-light==0.2.0"',
            (_REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(),
        )

    def test_release_wheel_discovery(self):
        for filename in ("ci_core.yml", "docs.yml", "build_release_wheel.yml"):
            with self.subTest(workflow=filename):
                self.assertIn(
                    "  PIP_FIND_LINKS: "
                    "https://github.com/xadupre/onnx-light/releases/expanded_assets/0.2.0",
                    (_REPO_ROOT / ".github" / "workflows" / filename).read_text(),
                )

    def test_supported_python_versions(self):
        project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(project["requires-python"], ">=3.12")
        self.assertIn(
            'python-version: ["3.12", "3.13"]',
            (_REPO_ROOT / ".github" / "workflows" / "build_release_wheel.yml").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
