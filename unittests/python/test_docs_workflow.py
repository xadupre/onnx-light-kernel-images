# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Check documentation scheduling and failure reporting."""

import unittest
from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "docs.yml"


class TestDocsWorkflow(unittest.TestCase):
    def test_weekly_only(self):
        workflow = _WORKFLOW.read_text()
        triggers = workflow.split("\non:\n", 1)[1].split("\nconcurrency:", 1)[0]
        self.assertIn("  workflow_dispatch:", triggers)
        self.assertIn('    - cron: "0 6 * * 1"', triggers)
        self.assertNotIn("  push:", triggers)
        self.assertNotIn("  pull_request:", triggers)
        self.assertIn("run: sphinx-build -b html docs dist/html", workflow)

    def test_failure_reporting(self):
        workflow = _WORKFLOW.read_text()
        build, report = workflow.split("\n  report_failure:\n", 1)
        self.assertNotIn("issues: write", build)
        self.assertIn("    needs: docs", report)
        self.assertIn("if: ${{ always() && needs.docs.result == 'failure' }}", report)
        self.assertIn("      issues: write", report)
        self.assertIn("GH_TOKEN: ${{ github.token }}", report)
        self.assertIn("GH_REPO: ${{ github.repository }}", report)
        self.assertIn("/actions/runs/${{ github.run_id }}", report)
        self.assertIn("gh issue list --state open", report)
        self.assertIn('gh issue comment "$issue" --body "$body"', report)
        self.assertIn('gh issue create --title "$title" --body "$body"', report)

    def test_full_runtime_without_build_isolation(self):
        workflow = _WORKFLOW.read_text()
        runtime = workflow.index("- name: Install onnx-light Python runtime")
        dependencies = workflow.index("- name: Install Python build dependencies")
        build = workflow.index("- name: Build and install package with docs extras")
        docs = workflow.index("- name: Build documentation")
        self.assertLess(runtime, dependencies)
        self.assertLess(dependencies, build)
        self.assertLess(build, docs)
        self.assertIn(
            "https://github.com/xadupre/onnx-light/releases/download/${ONNX_LIGHT_VERSION}/"
            "onnx_light-${ONNX_LIGHT_VERSION}-cp313-cp313-"
            "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            workflow[runtime:dependencies],
        )
        self.assertIn('Path(get_cpp_build_info()["library_dir"])', workflow[runtime:dependencies])
        self.assertIn(
            'library_dir.parent.parent / "onnx_light.libs"', workflow[runtime:dependencies]
        )
        self.assertIn('os.environ.get("LD_LIBRARY_PATH", "")', workflow[runtime:dependencies])
        self.assertIn(
            'env.write(f"LD_LIBRARY_PATH={os.pathsep.join(paths)}\\n")',
            workflow[runtime:dependencies],
        )
        for requirement in ('"cmake>=4.0"', '"scikit-build-core>=0.3.3"', '"nanobind>=1.3.2"'):
            self.assertIn(requirement, workflow[dependencies:build])
        self.assertIn("pip install --no-build-isolation", workflow[build:docs])
        self.assertIn(".[docs]", workflow[build:docs])


if __name__ == "__main__":
    unittest.main()
