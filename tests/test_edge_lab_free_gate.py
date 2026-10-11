"""Safety and output contract tests for the free research-only pregame gate."""
import unittest
from scripts.edge_lab_free_gate import build_report


class TestEdgeLabFreeGate(unittest.TestCase):
    def test_blocked_projection_stays_blocked(self):
        row = build_report({"status": "BLOCKED", "reason": "no data"}, "2026_05_BUF_LA")
        self.assertEqual(row["status"], "BLOCKED")
        self.assertFalse(row["bet_authorized"])

    def test_successful_projection_is_not_approved_bet(self):
        games = [{"game_id": "2026_05_BUF_LA", "home_team": "LA", "away_team": "BUF",
                  "model_margin_home": -0.5, "model_total": 50.0}]
        row = build_report({"status": "READY", "games": games}, "2026_05_BUF_LA")
        self.assertEqual(row["status"], "RESEARCH_ONLY")
        self.assertEqual(row["decision"], "PASS")
        self.assertIsNone(row["estimated_ev"])
        self.assertEqual(row["projection"]["model_implied_away_points"], 25.25)

    def test_nonfinite_projection_fails_closed(self):
        games = [{"game_id": "2026_05_BUF_LA", "model_margin_home": "nan",
                  "model_total": 50.0}]
        row = build_report({"status": "READY", "games": games}, "2026_05_BUF_LA")
        self.assertEqual(row["status"], "BLOCKED")

    def test_missing_game_fails_closed(self):
        row = build_report({"status": "READY", "games": []}, "2026_05_BUF_LA")
        self.assertEqual(row["status"], "BLOCKED")

    def test_duplicate_game_fails_closed(self):
        x = {"game_id": "2026_05_BUF_LA", "model_margin_home": 0.0,
             "model_total": 50.0}
        row = build_report({"status": "READY", "games": [x, x]}, "2026_05_BUF_LA")
        self.assertEqual(row["status"], "BLOCKED")

    def test_bool_not_accepted_as_score(self):
        x = {"game_id": "2026_05_BUF_LA", "model_margin_home": True,
             "model_total": 50.0}
        row = build_report({"status": "READY", "games": [x]}, "2026_05_BUF_LA")
        self.assertEqual(row["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
