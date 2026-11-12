# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Check the documentation homepage's build metadata."""

import runpy
import sys
import unittest
from pathlib import Path

from onnx_light_kernel_images import __version__

_DOCS = Path(__file__).resolve().parents[2] / "docs"


class TestDocsHomepage(unittest.TestCase):
    def test_documented_version(self):
        original_path = sys.path.copy()
        self.addCleanup(setattr, sys, "path", original_path)
        config = runpy.run_path(str(_DOCS / "conf.py"))
        self.assertEqual(config["release"], __version__)
        self.assertIn("Version: |release|", (_DOCS / "index.rst").read_text())

    def test_generation_date(self):
        self.assertIn("Last generated: |today|", (_DOCS / "index.rst").read_text())


if __name__ == "__main__":
    unittest.main()
