import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import benchmark_exact


class SampleBenchmarkTests(unittest.TestCase):
    def run_samples(self, slowdown):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "sample/input"
            inputs.mkdir(parents=True)
            for name in ["a.png", "b.jpg", "c.jpeg", "cliparts-6x6.png", "ignored.txt"]:
                (inputs / name).touch()
            for name in ["baseline", "candidate"]:
                (root / name).touch()
            calls = []
            digest = hashlib.sha256(b"reference svg").hexdigest()

            def run(binary, case, output, threads, timeout):
                calls.append((binary.name, case, benchmark_exact.CASES[case]))
                return {"seconds": 10 + (slowdown if binary.name == "candidate" else 0),
                        "sha256": digest, "summary": {}, "stages": {}}

            def git(command, **kwargs):
                return "revision\n" if command[1] == "rev-parse" else b"reference svg"

            argv = ["benchmark_exact.py", "--baseline", str(root / "baseline"),
                    "--candidate", str(root / "candidate"), "--output-dir", str(root / "results"),
                    "--samples", "--repeats", "2", "--threads", "0",
                    "--max-regression-seconds", "5"]
            error = None
            with (patch.object(benchmark_exact, "ROOT", root),
                  patch.dict(benchmark_exact.CASES, {}, clear=True),
                  patch.object(benchmark_exact, "run", side_effect=run),
                  patch.object(benchmark_exact.subprocess, "check_output", side_effect=git),
                  patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO())):
                try:
                    benchmark_exact.main()
                except SystemExit as exception:
                    error = str(exception)
            report = json.loads((root / "results/results.json").read_text())
            return calls, report, error

    def test_discovers_all_images_and_checks_both_execution_orders(self):
        calls, report, error = self.run_samples(4.999)
        self.assertIsNone(error)
        self.assertEqual(len(calls), 16)
        self.assertEqual([c[0] for c in calls[:2]], ["baseline", "candidate"])
        self.assertEqual([c[0] for c in calls[8:10]], ["candidate", "baseline"])
        self.assertEqual(len(report["committed_svg_sha256"]), 4)
        self.assertTrue(report["all_outputs_equal"])
        self.assertEqual(report["regressions"], [])
        key_calls = [c for c in calls if c[1] == "sample-cliparts-6x6.png"]
        self.assertEqual(key_calls[0][2][1], ["--remove-chroma-key-background"])

    def test_rejects_five_seconds_without_averaging_it_away(self):
        calls, report, error = self.run_samples(5)
        self.assertIn("Rejected:", error)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["regressions"][0]["seconds"], 5)
        self.assertNotIn("all_outputs_equal", report)


if __name__ == "__main__":
    unittest.main()
