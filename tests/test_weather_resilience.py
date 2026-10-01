from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.error import URLError

import pytest

from nfl.weather import OpenMeteoNFLWeather


def test_known_neutral_venue_does_not_require_geocoder(tmp_path) -> None:
    def forbidden_fetch(_: str) -> dict[str, object]:
        raise AssertionError("known NFL neutral venue should not require geocoding")

    client = OpenMeteoNFLWeather(
        cache_path=tmp_path / "geo.json",
        fetch_json=forbidden_fetch,
    )
    coords = client.venue_coordinates(
        home_team="WAS",
        stadium="Tottenham Hotspur Stadium",
        neutral=True,
    )

    assert coords == pytest.approx((51.6043, -0.0665))


def test_forecast_request_is_scoped_to_kickoff_day(tmp_path) -> None:
    seen: list[str] = []

    def fake_fetch(url: str) -> dict[str, object]:
        seen.append(url)
        return {
            "hourly": {
                "time": ["2026-10-04T13:00", "2026-10-04T14:00"],
                "temperature_2m": [55.0, 56.0],
                "precipitation_probability": [20.0, 25.0],
                "wind_speed_10m": [8.0, 9.0],
                "wind_gusts_10m": [12.0, 13.0],
            }
        }

    client = OpenMeteoNFLWeather(
        cache_path=tmp_path / "geo.json",
        fetch_json=fake_fetch,
    )
    result = client.forecast(
        kickoff=datetime(2026, 10, 4, 13, 30, tzinfo=UTC),
        latitude=51.6043,
        longitude=-0.0665,
    )

    assert result["weather_source"] == "Open-Meteo"
    assert result["forecast_time"] in {"2026-10-04T13:00", "2026-10-04T14:00"}
    assert len(seen) == 1
    assert "start_date=2026-10-04" in seen[0]
    assert "end_date=2026-10-04" in seen[0]
    assert "forecast_days" not in seen[0]


def test_http_json_retries_transient_url_errors(monkeypatch, tmp_path) -> None:
    attempts = 0

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"ok": True}).encode()

    def fake_open(*_: object, **__: object) -> Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise URLError("temporary TLS failure")
        return Response()

    monkeypatch.setattr("nfl.weather.urlopen", fake_open)
    monkeypatch.setattr("nfl.weather.time.sleep", lambda _: None)
    client = OpenMeteoNFLWeather(
        cache_path=tmp_path / "geo.json",
        max_attempts=3,
    )

    assert client._http_json("https://example.test/weather") == {"ok": True}
    assert attempts == 3
