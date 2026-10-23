# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Exercise the release steps with an explicit GitHub CLI test double."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash"), "a POSIX shell is required for the CLI double"
)
class TestReleaseWorkflows(unittest.TestCase):
    def test_release_notes_and_uploads(self):
        for workflow, asset in (
            ("build_release_cpp.yml", "onnx-light-kernel-images-cpp-"),
            ("build_release_wheel.yml", "./wheelhouse/*.whl"),
        ):
            path = _REPO_ROOT / ".github" / "workflows" / workflow
            script = textwrap.dedent(path.read_text().rsplit("        run: |\n", 1)[1])
            # Substitute the Actions expressions before running the shell step.
            script = (
                script.replace("${{ github.workspace }}", "/tmp/release-test")
                .replace("${{ matrix.artifact_suffix }}", "linux-x86_64")
                .replace("${{ matrix.archive_ext }}", "tar.gz")
            )
            for state, input_tag in (
                ("missing", ""),
                ("existing", "refs/tags/0.1.0"),
                ("concurrent", "0.1.0"),
                ("failure", ""),
            ):
                with (
                    self.subTest(workflow=workflow, state=state),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    gh = root / "gh"
                    gh.write_text(
                        f"#!{sys.executable}\n"
                        "import json, os, sys\n"
                        "from pathlib import Path\n"
                        "log = Path(os.environ['GH_TEST_LOG'])\n"
                        "calls = log.read_text().splitlines() if log.exists() else []\n"
                        "with log.open('a') as stream:\n"
                        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                        "state = os.environ['GH_TEST_STATE']\n"
                        "command = sys.argv[2]\n"
                        "if command == 'view':\n"
                        "    sys.exit(0 if state == 'existing' or "
                        "(state == 'concurrent' and calls) else 1)\n"
                        "if command == 'create':\n"
                        "    sys.exit(1 if state in ('concurrent', 'failure') else 0)\n"
                    )
                    gh.chmod(0o755)
                    log = root / "calls.jsonl"
                    result = subprocess.run(
                        ["bash", "-e", "-c", script],
                        env={
                            **os.environ,
                            "PATH": directory + os.pathsep + os.environ["PATH"],
                            "GH_TEST_LOG": str(log),
                            "GH_TEST_STATE": state,
                            "INPUT_TAG": input_tag,
                            "GITHUB_REF_NAME": "0.1.0",
                        },
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    if state == "failure":
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse(any(call[1] == "upload" for call in calls))
                        continue
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(all(call[2] == "0.1.0" for call in calls))
                    if state == "existing":
                        self.assertFalse(any(call[1] == "create" for call in calls))
                    else:
                        create = next(call for call in calls if call[1] == "create")
                        self.assertIn("--generate-notes", create)
                        self.assertIn("--verify-tag", create)
                    self.assertEqual(calls[-1][:3], ["release", "upload", "0.1.0"])
                    self.assertIn(asset, calls[-1][3])
                    self.assertIn("--clobber", calls[-1])


if __name__ == "__main__":
    unittest.main()
