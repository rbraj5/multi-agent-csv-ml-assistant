from __future__ import annotations

import unittest
from pathlib import Path


REQUIRED_HEADINGS = [
    "## Overview",
    "## Production Use Case",
    "## Architecture",
    "## LangGraph Workflow",
    "## API Usage",
    "## Docker Run",
    "## Local Streamlit/CLI Demo",
    "## Configuration",
    "## Testing",
    "## Azure Container Apps Deployment Path",
    "## Production Readiness Notes",
    "## Limitations and Next Steps",
]

OVERCLAIMS = ["deployed to azure", "running in production", "production deployed"]


class ReadmeContractTest(unittest.TestCase):
    def test_required_headings_exist_in_order(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        positions = []
        for heading in REQUIRED_HEADINGS:
            position = readme.find(heading)
            self.assertNotEqual(position, -1, f"Missing heading: {heading}")
            positions.append(position)
        self.assertEqual(positions, sorted(positions), "README headings are out of order")

    def test_readme_does_not_overclaim_deployment(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8").lower()
        for phrase in OVERCLAIMS:
            self.assertNotIn(phrase, readme)


if __name__ == "__main__":
    unittest.main()
