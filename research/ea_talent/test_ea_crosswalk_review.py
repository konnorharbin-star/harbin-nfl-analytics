import csv
import tempfile
import unittest
from pathlib import Path

from ea_crosswalk_review import review_candidates


class ReviewTests(unittest.TestCase):
    def test_suggestions_are_never_approved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            a = p / "ea.csv"
            b = p / "roster.csv"
            o = p / "review.csv"
            a.write_text("ea_player_id,player_name,team,position\ne1,Test Quarterback,T,QB\n")
            b.write_text("model_player_id,player_name,team,position\nm1,Test Quarterback,T,QB\n")
            result = review_candidates(a, b, o)
            self.assertEqual(result["automatically_approved"], 0)
            with o.open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["suggested_model_player_id"], "m1")
            self.assertEqual(row["review_decision"], "UNREVIEWED")
            self.assertEqual(row["verified_model_player_id"], "")


if __name__ == "__main__":
    unittest.main()
