import csv
import tempfile
import unittest
from pathlib import Path
from ea_player_talent import load_snapshot, team_features, matchup_features

class EATalentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ea.csv"
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["player_id","team","position","ovr","snapshot_at","available"])
            w.writeheader()
            for team in ("HOME","AWAY"):
                for i, p in enumerate(["QB","QB","WR","WR","WR","WR","WR"]):
                    w.writerow(dict(player_id=f"{team}-{i}",team=team,position=p,ovr=90 if team=="HOME" else 80,
                                    snapshot_at="2026-08-01T00:00:00+00:00",available="0" if team=="HOME" and i==0 else "1"))
    def tearDown(self):
        self.tmp.cleanup()
    def test_injury_replacement_and_missing_coverage(self):
        rows=load_snapshot(self.path,prediction_at="2026-09-01T12:00:00+00:00",kickoff_at="2026-09-01T20:00:00+00:00")
        feats=team_features(rows)
        self.assertEqual(feats["HOME"]["qb_talent"],90)
        self.assertEqual(feats["HOME"]["skill_talent"],90)
        self.assertIsNone(feats["HOME"]["ol_talent"])
        self.assertEqual(matchup_features(feats,"HOME","AWAY")["qb_talent_diff"],10)
    def test_future_snapshot_blocked(self):
        with self.assertRaises(ValueError):
            load_snapshot(self.path,prediction_at="2026-07-01T12:00:00+00:00",kickoff_at="2026-09-01T20:00:00+00:00")
    def test_post_kickoff_blocked(self):
        with self.assertRaises(ValueError):
            load_snapshot(self.path,prediction_at="2026-09-02T12:00:00+00:00",kickoff_at="2026-09-01T20:00:00+00:00")

if __name__=="__main__":
    unittest.main()
