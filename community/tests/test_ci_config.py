import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI_FILES = (
    ".gitlab-ci.yml",
    ".github/workflows/pages.yml",
    "community/ci/github-actions.yml",
)


class CiConfigTests(unittest.TestCase):
    def test_site_and_community_jobs_never_sync_the_adapter_environment(self):
        # The site and community tooling needs the standard library plus Markdown.
        # Without --no-project, uv first installs the whole adapter (pandas, pyarrow,
        # scipy, the Brightway packages) into the job just to render Markdown.
        commands = []
        for name in CI_FILES:
            for number, line in enumerate((ROOT / name).read_text().splitlines(), 1):
                if "uv run" in line and not line.lstrip().startswith("#"):
                    commands.append((name, number, line.strip()))
        self.assertTrue(commands)
        offenders = [c for c in commands if "--no-project" not in c[2]]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
