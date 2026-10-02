from __future__ import annotations

import json

from nfl.research_status import write_research_status


def main() -> None:
    report = write_research_status()
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
