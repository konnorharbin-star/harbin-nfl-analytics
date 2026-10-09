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
            "player_id,team,position,ovr,snapshot_at\n"
            "a,A,QB,90,2026-08-01T00:00:00+00:00\n"
            "b,A,QB,70,2026-08-01T00:00:00+00:00\n"
            "c,A,QB,65,2026-08-01T00:00:00+00:00\n"
        )
        self.lineup.write_text(
            "player_id,team,role,depth_rank,observed_at,obtained_at,source_url,verified\n"
            "a,A,out,1,2026-09-01T10:00:00+00:00,"
            "2026-09-01T10:01:00+00:00,https://example.com/lineup,true\n"
            "b,A,starter,2,2026-09-01T10:00:00+00:00,"
            "2026-09-01T10:01:00+00:00,https://example.com/lineup,true\n"
            "c,A,reserve,3,2026-09-01T10:00:00+00:00,"
            "2026-09-01T10:01:00+00:00,https://example.com/lineup,true\n"
        )
        self.games.write_text(
            "game_id,season,kickoff_at,forecast_at,"
            "base_margin,candidate_margin,actual_margin\n"
            "g1,2025,2025-09-01T20:00:00+00:00,"
            "2025-09-01T10:00:00+00:00,4,3,2\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def assess(self):
        return lineup_units(
            self.ratings,
            self.lineup,
            prediction_at="2026-09-01T12:00:00+00:00",
            kickoff_at="2026-09-01T20:00:00+00:00",
        )[("A", "QB")]

    def test_true_replacement_not_third_string(self):
        qb = self.assess()
        self.assertTrue(qb["qb_ready"])
        self.assertEqual(qb["qb_replacement_player_id"], "b")
        self.assertEqual(qb["unavailable_vs_reserve_gap"], 20.0)
        self.assertEqual(qb["qb_gap_reason"], "CONFIRMED_QB1_OUT_REPLACEMENT")

    def test_healthy_qb_one_is_zero_gap(self):
        s = self.lineup.read_text()
        self.lineup.write_text(s.replace("a,A,out,1,", "a,A,starter,1,").replace(
            "b,A,starter,2,", "b,A,reserve,2,"
        ))
        qb = self.assess()
        self.assertTrue(qb["qb_ready"])
        self.assertEqual(qb["unavailable_vs_reserve_gap"], 0.0)

    def test_qb_one_uncertain_blocks_gap(self):
        s = self.lineup.read_text().replace("a,A,out,1,", "a,A,uncertain,1,")
        self.lineup.write_text(s)
        qb = self.assess()
        self.assertFalse(qb["qb_ready"])
        self.assertIsNone(qb["unavailable_vs_reserve_gap"])

    def test_missing_qb_one_blocks_gap(self):
        s = self.lineup.read_text()
        self.lineup.write_text(
            "\n".join(line for line in s.splitlines() if not line.startswith("a,A,")) + "\n"
        )
        qb = self.assess()
        self.assertFalse(qb["qb_ready"])
        self.assertIsNone(qb["unavailable_vs_reserve_gap"])

    def test_future_acquisition_blocked(self):
        s = self.lineup.read_text().replace(
            "2026-09-01T10:01:00+00:00", "2026-09-01T13:01:00+00:00"
        )
        self.lineup.write_text(s)
        with self.assertRaisesRegex(ValueError, "Future"):
            self.assess()

    def test_unverified_source_blocked(self):
        s = self.lineup.read_text().replace(
            "https://example.com/lineup,true", "https://example.com/lineup,false"
        )
        self.lineup.write_text(s)
        with self.assertRaisesRegex(ValueError, "source-verified"):
            self.assess()

    def test_duplicate_rank_blocked(self):
        s = self.lineup.read_text().replace("b,A,starter,2,", "b,A,starter,1,")
        self.lineup.write_text(s)
        with self.assertRaisesRegex(ValueError, "Ambiguous depth"):
            self.assess()

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

    def test_stale_assignment_blocks_gap(self):
        s = self.lineup.read_text().replace(
            "2026-09-01T10:00:00+00:00", "2026-08-20T10:00:00+00:00"
        )
        self.lineup.write_text(s)
        qb = self.assess()
        self.assertFalse(qb["qb_ready"])
        self.assertIsNone(qb["unavailable_vs_reserve_gap"])
        self.assertEqual(qb["qb_gap_reason"], "STALE_QB_ASSIGNMENT")

    def test_named_third_string_starter_uses_his_rating(self):
        s = self.lineup.read_text().replace(
            "b,A,starter,2,", "b,A,reserve,2,"
        ).replace("c,A,reserve,3,", "c,A,starter,3,")
        self.lineup.write_text(s)
        qb = self.assess()
        self.assertTrue(qb["qb_ready"])
        self.assertEqual(qb["qb_replacement_player_id"], "c")
        self.assertEqual(qb["unavailable_vs_reserve_gap"], 25.0)


if __name__ == "__main__":
    unittest.main()
