import json
import tempfile
import unittest
from pathlib import Path

from ea_evidence_gate import evaluate_manifest


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "manifest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_evidence_blocks(self):
        self.path.write_text("{}")
        result = evaluate_manifest(self.path)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["production_weight"], 0)
        self.assertIsNone(result["metrics"])
        self.assertEqual(len(result["missing"]), 5)

    def test_unverified_files_do_not_pass(self):
        self.path.write_text(
            json.dumps({"ratings_archive": {"verified": True, "path": "/does/not/exist"}})
        )
        self.assertFalse(evaluate_manifest(self.path)["gates"]["ratings_archive"])


if __name__ == "__main__":
    unittest.main()
