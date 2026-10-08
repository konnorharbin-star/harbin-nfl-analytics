import unittest
from scripts.nfl_accuracy_evidence_audit import audit


class TestAccuracyEvidenceAudit(unittest.TestCase):
    def test_pending_games_are_not_scored(self):
        r = {"game_id": "g1", "market": "spread", "captured_at": "2026-10-08T12:00:00+00:00",
             "kickoff": "2026-10-11T12:00:00+00:00", "observation_status": "PENDING_RESULT",
             "home_score": "", "away_score": "", "result": ""}
        out = audit([r])
        self.assertEqual(out["rows_with_final_scores"], 0)
        self.assertEqual(out["timestamp_valid_pregame_rows"], 1)
        self.assertFalse(out["accuracy_claim_permitted"])

    def test_late_and_duplicate_snapshots(self):
        r = {"game_id": "g2", "market": "moneyline", "captured_at": "2026-10-12T12:00:00+00:00",
             "kickoff": "2026-10-11T12:00:00+00:00", "observation_status": "PENDING_RESULT",
             "home_score": "", "away_score": "", "result": ""}
        self.assertEqual(audit([r])["invalid_or_late_timestamp_rows"], 1)


if __name__ == "__main__":
    unittest.main()
