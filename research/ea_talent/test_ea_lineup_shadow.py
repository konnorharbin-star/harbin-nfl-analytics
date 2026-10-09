import tempfile
import unittest
from pathlib import Path

from ea_lineup_shadow import lineup_units, shadow_metrics


class Phase3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = Path(self.tmp.name)
        self.ratings, self.lineup, self.games = [
            p / n for n in ("ratings.csv", "lineup.csv", "games.csv")
        ]
        self.ratings.write_text(
            "player_id,team,position,ovr,snapshot_at\na,A,QB,90,2026-08-01T00:00:00+00:00\nb,A,QB,70,2026-08-01T00:00:00+00:00\nc,A,QB,65,2026-08-01T00:00:00+00:00\n"
        )
        self.lineup.write_text(
            "player_id,team,role,observed_at\na,A,out,2026-09-01T10:00:00+00:00\nb,A,starter,2026-09-01T10:00:00+00:00\nc,A,reserve,2026-09-01T10:00:00+00:00\n"
        )
        self.games.write_text(
            "game_id,season,kickoff_at,forecast_at,base_margin,candidate_margin,actual_margin\ng1,2025,2025-09-01T20:00:00+00:00,2025-09-01T10:00:00+00:00,4,3,2\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_lineup(self):
        out = lineup_units(
            self.ratings,
            self.lineup,
            prediction_at="2026-09-01T12:00:00+00:00",
            kickoff_at="2026-09-01T20:00:00+00:00",
        )
        self.assertTrue(out[("A", "QB")]["qb_ready"])
        self.assertEqual(out[("A", "QB")]["unavailable_vs_reserve_gap"], 25)

    def test_future_lineup_blocked(self):
        with self.assertRaises(ValueError):
            lineup_units(
                self.ratings,
                self.lineup,
                prediction_at="2026-09-01T09:00:00+00:00",
                kickoff_at="2026-09-01T20:00:00+00:00",
            )

    def test_shadow(self):
        out = shadow_metrics(self.games)
        self.assertEqual(out["overall"]["base_mae"], 2)
        self.assertEqual(out["overall"]["candidate_mae"], 1)

    def test_shadow_future_blocked(self):
        self.games.write_text(
            self.games.read_text().replace("2025-09-01T10:00:00", "2025-09-02T10:00:00")
        )
        with self.assertRaises(ValueError):
            shadow_metrics(self.games)


if __name__ == "__main__":
    unittest.main()
