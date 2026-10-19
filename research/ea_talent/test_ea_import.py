import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ea_import import normalize


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        p = Path(self.temp.name)
        self.raw, self.mapping, self.meta, self.out = [
            p / n for n in ("raw.csv", "map.csv", "meta.json", "normalized.csv")
        ]
        self.raw.write_text(
            "ea_player_id,player_name,team,position,ovr,awr\nEA1,Sample Player,AAA,QB,90,89\n",
            encoding="utf-8",
        )
        self.mapping.write_text(
            "ea_player_id,model_player_id,team,effective_from,effective_to\nEA1,MODEL1,AAA,2026-01-01T00:00:00+00:00,\n",
            encoding="utf-8",
        )
        self.meta.write_text(
            json.dumps(
                {
                    "source_url": "https://example.org/export",
                    "snapshot_at": "2026-08-01T00:00:00+00:00",
                    "obtained_at": "2026-08-02T00:00:00+00:00",
                    "source_sha256": hashlib.sha256(self.raw.read_bytes()).hexdigest(),
                    "license_note": "fixture only",
                }
            )
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_import_and_no_overwrite(self):
        result = normalize(
            self.raw,
            self.mapping,
            self.meta,
            self.out,
            as_of="2026-09-01T00:00:00+00:00",
            min_coverage=1,
        )
        self.assertEqual(result["matched_players"], 1)
        self.assertIn("MODEL1", self.out.read_text())
        with self.assertRaises(FileExistsError):
            normalize(
                self.raw, self.mapping, self.meta, self.out, as_of="2026-09-01T00:00:00+00:00"
            )

    def test_future_snapshot_fails(self):
        with self.assertRaises(ValueError):
            normalize(
                self.raw, self.mapping, self.meta, self.out, as_of="2026-07-01T00:00:00+00:00"
            )

    def test_ambiguous_crosswalk_fails(self):
        with self.mapping.open("a") as f:
            f.write("EA1,MODEL2,AAA,2026-01-01T00:00:00+00:00,\n")
        with self.assertRaises(ValueError):
            normalize(
                self.raw, self.mapping, self.meta, self.out, as_of="2026-09-01T00:00:00+00:00"
            )

    def test_digest_fails(self):
        self.raw.write_text(self.raw.read_text() + "bad")
        with self.assertRaises(ValueError):
            normalize(
                self.raw, self.mapping, self.meta, self.out, as_of="2026-09-01T00:00:00+00:00"
            )


if __name__ == "__main__":
    unittest.main()
