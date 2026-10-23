# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Check release metadata and the matching onnx-light runtime dependency."""

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
        self.assertIn(
            '"onnx-light==0.2.0"',
            (_REPO_ROOT / "pyproject.toml").read_text(),
        )
        self.assertIn(
            'pip install "onnx-light==0.2.0"',
            (_REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
