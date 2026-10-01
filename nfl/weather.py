"""Free current NFL venue/weather context using deterministic team locations and Open-Meteo."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .contracts import DataContractError

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
NFL_SCHEDULE_TIMEZONE = ZoneInfo("America/New_York")

# Approximate home-stadium coordinates and local timezone. These values are context
# metadata only; they are not learned model features and never enter the fair-score fit.
TEAM_HOME: dict[str, tuple[float, float, str]] = {
    "ARI": (33.5276, -112.2626, "America/Phoenix"),
    "ATL": (33.7554, -84.4008, "America/New_York"),
    "BAL": (39.2780, -76.6227, "America/New_York"),
    "BUF": (42.7738, -78.7868, "America/New_York"),
    "CAR": (35.2258, -80.8528, "America/New_York"),
    "CHI": (41.8623, -87.6167, "America/Chicago"),
    "CIN": (39.0955, -84.5161, "America/New_York"),
    "CLE": (41.5061, -81.6995, "America/New_York"),
    "DAL": (32.7473, -97.0945, "America/Chicago"),
    "DEN": (39.7439, -105.0201, "America/Denver"),
    "DET": (42.3400, -83.0456, "America/Detroit"),
    "GB": (44.5013, -88.0622, "America/Chicago"),
    "HOU": (29.6847, -95.4107, "America/Chicago"),
    "IND": (39.7601, -86.1639, "America/Indiana/Indianapolis"),
    "JAX": (30.3239, -81.6373, "America/New_York"),
    "KC": (39.0489, -94.4839, "America/Chicago"),
    "LA": (33.9535, -118.3392, "America/Los_Angeles"),
    "LAC": (33.9535, -118.3392, "America/Los_Angeles"),
    "LV": (36.0908, -115.1830, "America/Los_Angeles"),
    "MIA": (25.9580, -80.2389, "America/New_York"),
    "MIN": (44.9738, -93.2581, "America/Chicago"),
    "NE": (42.0909, -71.2643, "America/New_York"),
    "NO": (29.9511, -90.0812, "America/Chicago"),
    "NYG": (40.8135, -74.0745, "America/New_York"),
    "NYJ": (40.8135, -74.0745, "America/New_York"),
    "PHI": (39.9008, -75.1675, "America/New_York"),
    "PIT": (40.4468, -80.0158, "America/New_York"),
    "SEA": (47.5952, -122.3316, "America/Los_Angeles"),
    "SF": (37.4030, -121.9700, "America/Los_Angeles"),
    "TB": (27.9759, -82.5033, "America/New_York"),
    "TEN": (36.1665, -86.7713, "America/Chicago"),
    "WAS": (38.9078, -76.8645, "America/New_York"),
}


def is_indoor_roof(value: object) -> bool:
    roof = "" if value is None else str(value).strip().lower()
    return any(token in roof for token in ("dome", "indoor", "closed"))


def kickoff_utc(gameday: object, gametime: object) -> datetime | None:
    if gameday is None or gametime in {None, ""}:
        return None
    try:
        local = datetime.fromisoformat(f"{gameday}T{gametime}")
    except ValueError:
        return None
    if local.tzinfo is None:
        local = local.replace(tzinfo=NFL_SCHEDULE_TIMEZONE)
    return local.astimezone(UTC)


def haversine_miles(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius = 3958.7613
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))


def team_travel_context(away_team: str, home_team: str) -> dict[str, object]:
    away = TEAM_HOME.get(away_team)
    home = TEAM_HOME.get(home_team)
    if away is None or home is None:
        return {
            "away_travel_miles": None,
            "home_travel_miles": 0.0 if home is not None else None,
            "timezone_shift_hours": None,
            "travel_source_available": False,
        }
    miles = haversine_miles(away[0], away[1], home[0], home[1])
    reference = datetime(2026, 10, 1, 12, tzinfo=UTC)
    away_offset = reference.astimezone(ZoneInfo(away[2])).utcoffset()
    home_offset = reference.astimezone(ZoneInfo(home[2])).utcoffset()
    shift = None
    if away_offset is not None and home_offset is not None:
        shift = abs((home_offset - away_offset).total_seconds() / 3600.0)
    return {
        "away_travel_miles": miles,
        "home_travel_miles": 0.0,
        "timezone_shift_hours": shift,
        "travel_source_available": True,
    }


def weather_risk(weather: dict[str, object], *, indoor: bool = False) -> float:
    if indoor:
        return 0.0
    try:
        wind = float(weather.get("wind_mph") or 0.0)
        gust = float(weather.get("wind_gust_mph") or 0.0)
        precip = float(weather.get("precip_probability") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    temp_raw = weather.get("temperature_f")
    temp_penalty = 0.0
    try:
        temp = float(temp_raw)
        if math.isfinite(temp):
            temp_penalty = max(0.0, (35.0 - temp) / 35.0) * 0.08
            temp_penalty += max(0.0, (temp - 95.0) / 25.0) * 0.05
    except (TypeError, ValueError):
        pass
    risk = max(0.0, (wind - 15.0) / 25.0) * 0.52
    risk += max(0.0, (precip - 40.0) / 60.0) * 0.25
    risk += max(0.0, (gust - 25.0) / 35.0) * 0.15
    risk += temp_penalty
    return max(0.0, min(1.0, risk))


class OpenMeteoNFLWeather:
    """Small cached Open-Meteo client for current outdoor NFL games."""

    def __init__(
        self,
        *,
        cache_path: str | Path = "data/cache/context/venue_geocodes.json",
        fetch_json: Callable[[str], dict[str, Any]] | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.timeout_seconds = float(timeout_seconds)
        self._fetch_json = fetch_json or self._http_json
        self._geocodes = self._load_cache()

    def _load_cache(self) -> dict[str, list[object]]:
        if not self.cache_path.exists():
            return {}
        try:
            value = json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._geocodes, indent=2, sort_keys=True))

    def _http_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": "harbin-nfl-analytics/0.1"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DataContractError(f"weather source request failed: {exc}") from exc

    def geocode(self, query: str) -> tuple[float, float] | None:
        key = query.strip().lower()
        cached = self._geocodes.get(key)
        if isinstance(cached, list) and len(cached) == 2:
            return float(cached[0]), float(cached[1])
        url = OPEN_METEO_GEOCODE + "?" + urlencode(
            {"name": query, "count": 1, "language": "en", "format": "json"}
        )
        payload = self._fetch_json(url)
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None
        first = results[0]
        if not isinstance(first, dict):
            return None
        try:
            coords = float(first["latitude"]), float(first["longitude"])
        except (KeyError, TypeError, ValueError):
            return None
        self._geocodes[key] = [coords[0], coords[1]]
        self._save_cache()
        return coords

    def venue_coordinates(
        self,
        *,
        home_team: str,
        stadium: object = None,
        neutral: bool = False,
    ) -> tuple[float, float] | None:
        if not neutral and home_team in TEAM_HOME:
            value = TEAM_HOME[home_team]
            return value[0], value[1]
        if stadium not in {None, ""}:
            return self.geocode(str(stadium))
        return None

    def forecast(
        self,
        *,
        kickoff: datetime,
        latitude: float,
        longitude: float,
    ) -> dict[str, object]:
        url = OPEN_METEO_FORECAST + "?" + urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": (
                    "temperature_2m,precipitation_probability,wind_speed_10m,"
                    "wind_gusts_10m"
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
                "forecast_days": 16,
            }
        )
        payload = self._fetch_json(url)
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise DataContractError("Open-Meteo response missing hourly data")
        times = hourly.get("time")
        if not isinstance(times, list) or not times:
            raise DataContractError("Open-Meteo response missing hourly timestamps")
        parsed: list[tuple[float, int]] = []
        for index, value in enumerate(times):
            try:
                stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            parsed.append((abs((stamp.astimezone(UTC) - kickoff.astimezone(UTC)).total_seconds()), index))
        if not parsed:
            raise DataContractError("Open-Meteo hourly timestamps could not be parsed")
        _, index = min(parsed)

        def value(name: str) -> object:
            values = hourly.get(name)
            return values[index] if isinstance(values, list) and index < len(values) else None

        return {
            "temperature_f": value("temperature_2m"),
            "precip_probability": value("precipitation_probability"),
            "wind_mph": value("wind_speed_10m"),
            "wind_gust_mph": value("wind_gusts_10m"),
            "forecast_time": times[index],
            "weather_source": "Open-Meteo",
        }
