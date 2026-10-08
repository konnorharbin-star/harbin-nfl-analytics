import unittest

from scripts.nfl_score_accuracy_diagnostic import analyze


def forecast(gid="a", captured="2026-10-01T10:00:00+00:00", margin="4", total="44"):
    return {"game_id": gid, "captured_at": captured, "kickoff": "2026-10-02T20:00:00+00:00",
            "model_margin_home": margin, "model_total": total}


def final(gid="a", home="24", away="17"):
    return {"game_id": gid, "home_score": home, "away_score": away}


class ScoreAccuracyTests(unittest.TestCase):
    def test_correct_home_margin_total_errors(self):
        r = analyze([forecast()], [final()])
        self.assertEqual(r["sample_games"], 1)
        self.assertEqual(r["margin"]["mae"], 3)
        self.assertEqual(r["total"]["mae"], 3)

    def test_exclude_late_and_missing_results(self):
        r = analyze([forecast(captured="2026-10-03T10:00:00+00:00"), forecast(gid="b")], [final()])
        self.assertEqual(r["sample_games"], 0)
        self.assertEqual(r["rejected"]["late_snapshot"], 1)

    def test_earliest_valid_snapshot_fixed(self):
        later = forecast(captured="2026-10-02T12:00:00+00:00", margin="7")
        r = analyze([later, forecast()], [final()])
        self.assertEqual(r["margin"]["mae"], 3)

    def test_conflicting_finals_excluded(self):
        r = analyze([forecast()], [final(), final(home="30")])
        self.assertEqual(r["sample_games"], 0)

    def test_no_nan_and_naive_timestamp(self):
        r = analyze([forecast(margin="nan"), forecast(gid="b", captured="2026-10-01T10:00:00")],
                    [final(), final(gid="b")])
        self.assertEqual(r["sample_games"], 0)


if __name__ == "__main__":
    unittest.main()
