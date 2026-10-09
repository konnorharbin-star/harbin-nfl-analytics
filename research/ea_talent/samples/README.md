# Limited official Madden NFL 27 QB source sample

These **two factual example records** were checked against EA's public player profile pages on October 9, 2026. They demonstrate that the page path supplies a persistent-looking EA player ID, player name, team, position and OVR; EA IDs are **not model/GSIS player IDs**. The field `historical_backfill_permitted=false` is deliberate.

- Dak Prescott: EA player page `/dak-prescott/17567`, listed as Dallas Cowboys QB, Week 4 OVR 93, AWR 94, AGI 81.
- Baker Mayfield: EA player page `/baker-mayfield/13117`, listed as Tampa Bay Buccaneers QB, Week 4 OVR 81, AWR 78, AGI 82.

The source page calls these "Week 4" ratings. That does **not** independently prove the page content, player attributes or publication timing before the Thursday Cowboys–Buccaneers kickoff. The entries were checked **after** that game and cannot be used to regrade or fit its pregame forecast. It is also not evidence of either QB's pregame health/confirmed starter status.

This is a tiny review sample, **not** the complete EA dataset, not a licensed bulk export and not a normalized historical snapshot. It deliberately omits `snapshot_at`; passing a guessed older time into `ea_import.py` would create false point-in-time evidence. Existing `ea_import.py` still requires a manually verified, effective-dated mapping to model player IDs, complete provenance, source digest and a pregame acquisition time.

College Football 27 individual-player tables were browsable but their rows were not available in the verified results used for this example. No college player ratings were invented.

Live prediction weights and bet recommendations remain unchanged.
