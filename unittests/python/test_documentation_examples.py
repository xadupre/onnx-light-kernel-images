# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Runs every gallery example under ``docs/examples/`` as a unit test.

Each ``docs/examples/plot_*.py`` file is executed in a subprocess and the test
fails if the example raises an exception. The examples build a one-node
``ImageDecoder`` ONNX model, register this package's image kernels with the
onnx-light dispatch table and run the model through onnx-light's
:class:`ReferenceEvaluator`, so they require the onnx-light Python package (with
the operator-kernel runtime) as well as :mod:`PIL` and :mod:`matplotlib`. When
any of those dependencies is missing the corresponding example is skipped.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "docs" / "examples"


def _missing_dependency() -> str | None:
    """Returns the name of the first missing example dependency, if any."""
    for module in ("onnx_light", "PIL", "matplotlib", "numpy"):
        if importlib.util.find_spec(module) is None:
            return module
    # The examples exercise the operator-kernel runtime, which is absent from
    # the reduced onnx-light build (ONNX_LIGHT_BUILD_KERNELS=OFF).
    if importlib.util.find_spec("onnx_light_kernel_images.onnx_py._imgpykernels") is None:
        return "onnx_light_kernel_images.onnx_py._imgpykernels"
    return None


class TestDocumentationExamples(unittest.TestCase):
    """Executes each gallery example and asserts it runs without error."""

    def run_example(self, path: Path) -> None:
        env = dict(os.environ)
        # Force a non-interactive matplotlib backend so ``plt.show()`` does not
        # try to open a window on a headless machine.
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = os.pathsep.join([str(_REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )
        proc = subprocess.run(
            [sys.executable, "-u", str(path)],
            capture_output=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            self.fail(
                "Example {!r} failed with exit code {}:\n{}\n{}".format(
                    path.name,
                    proc.returncode,
                    proc.stdout.decode("utf-8", errors="ignore"),
                    proc.stderr.decode("utf-8", errors="ignore"),
                )
            )

    @classmethod
    def add_test_methods(cls) -> None:
        if not _EXAMPLES_DIR.is_dir():
            return
        reason = _missing_dependency()
        for path in sorted(_EXAMPLES_DIR.glob("plot_*.py")):

            def _test(self, path=path):
                self.run_example(path)

            if reason is not None:
                _test = unittest.skip(f"missing dependency: {reason}")(_test)

            setattr(cls, f"test_{path.stem}", _test)


TestDocumentationExamples.add_test_methods()


if __name__ == "__main__":
    unittest.main(verbosity=2)
