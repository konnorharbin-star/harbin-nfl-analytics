"""Source-conditioned NFL research player-availability scenarios."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import math

SPEC='NFL_WEB_CONDITIONED_PLAYER_RESEARCH_V1'
PRIORITY={'out':0,'injured reserve':0,'doubtful':1,'questionable':2}
