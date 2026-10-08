import unittest

from scripts.summarize_benchmarks import ROOT, stale


class BenchmarkSummaryTests(unittest.TestCase):
    def test_summary_and_docs_match_reports(self) -> None:
        paths = [str(path.relative_to(ROOT)) for path in stale()]
        self.assertEqual(paths, [], "run: uv run scripts/summarize_benchmarks.py")
