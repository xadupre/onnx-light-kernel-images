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


if __name__ == "__main__":
    unittest.main()
