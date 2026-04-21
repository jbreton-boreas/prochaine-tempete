# -*- coding: utf-8 -*-
"""
fetch_data.py – Open-Meteo data fetching and ski-score computation.

Design decisions
----------------
* Resorts are grouped by their ``model`` field.  A single batch API call is
  made per model so that Open-Meteo can use the most appropriate NWP model
  for each geographic region (e.g. ``gem_seamless`` for Canada,
  ``gfs_seamless`` for the USA, ``meteofrance_seamless`` for the French
  Alps, ``icon_seamless`` for the rest of Europe / Caucasus / Turkey,
  ``jma_seamless`` for Japan, ``ecmwf_ifs025`` for the southern
  hemisphere).  Developers can override the model for any resort simply by
  editing the ``model`` field in resorts.py.

* The Open-Meteo historical archive API (ERA5 reanalysis) is used for
  season-to-date snowfall totals.  Because the archive lags by ~5 days we
  request data up to ``today - 1``; Open-Meteo returns whatever is
  available without raising an error.

* Northern-hemisphere and southern-hemisphere resorts are batched
  separately for the archive API because they have different ski-season
  start dates.

* Ski Score (0–100) formula
  --------------------------
  Each component is first scaled independently to [0, 100]:
    base_score     = min(snow_base_cm   / 150,  1) * 100  (150 cm base → 100)
    forecast_score = min(next_10_snow   / 60,   1) * 100  (60 cm/10d  → 100)
    season_score   = min(season_total   / 400,  1) * 100  (400 cm/season → 100)
    cold_score     = max(0, min(-min_temp / 15, 1)) * 100  (-15 °C → 100)

  Final: ski_score = round(base * 0.30 + forecast * 0.40 + season * 0.20 + cold * 0.10)
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_DAYS = 10  # days ahead (changeable)
PAST_DAYS = 7       # days of past data included in the forecast request


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value, default=0.0):
    """Return float(value) or default if value is NaN / None."""
    try:
        f = float(value)
        return default if (f != f) else f  # NaN != NaN
    except (TypeError, ValueError):
        return default


def get_season_start_date(lat: float, today: date) -> date:
    """Return the start of the current ski season for the given latitude."""
    year = today.year
    month = today.month
    if lat < -10:  # Southern Hemisphere
        return date(year, 6, 1) if month >= 6 else date(year - 1, 6, 1)
    else:           # Northern Hemisphere
        return date(year, 11, 1) if month >= 11 else date(year - 1, 11, 1)


def _make_client(cache_file: str):
    cache_session = requests_cache.CachedSession(cache_file, expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# ─────────────────────────────────────────────────────────────────────────────
# Forecast API
# ─────────────────────────────────────────────────────────────────────────────

def _set_forecast_defaults(resort: dict):
    resort.setdefault("snow_base_cm", 0)
    resort.setdefault("next_10_snow_cm", 0.0)
    resort.setdefault("recent_7day_snow_cm", 0.0)
    resort.setdefault("min_10day_temp", None)
    resort.setdefault("max_10day_temp", None)
    resort.setdefault("avg_10day_temp", None)
    resort.setdefault("forecast_dates", [])
    resort.setdefault("forecast_daily_snow", [])
    resort.setdefault("snow_depth_dates", [])
    resort.setdefault("snow_depth_values", [])
    resort.setdefault("snow_depth_now_idx", 0)


def _process_forecast_response(resort: dict, response, now_utc: datetime, today_utc: date):
    # ── Hourly: snow depth ────────────────────────────────────────────────
    hourly = response.Hourly()
    hourly_depth_m = hourly.Variables(0).ValuesAsNumpy()
    hourly_dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    # Current snow base: last non-NaN value at or before now
    snow_base_m = 0.0
    for j, dt in enumerate(hourly_dates):
        if dt.to_pydatetime() <= now_utc:
            v = _safe_float(hourly_depth_m[j], snow_base_m)
            snow_base_m = v
    resort["snow_base_cm"] = round(snow_base_m * 100)

    # Snow-depth chart window: past PAST_DAYS + next FORECAST_DAYS
    window_start = now_utc - timedelta(days=PAST_DAYS)
    window_end = now_utc + timedelta(days=FORECAST_DAYS)
    depth_dates, depth_values = [], []
    now_idx = 0
    for j, dt in enumerate(hourly_dates):
        pdt = dt.to_pydatetime()
        if window_start <= pdt <= window_end:
            depth_dates.append(dt.strftime("%b %d %H:%M"))
            raw = hourly_depth_m[j]
            depth_values.append(
                round(float(raw) * 100, 1) if not pd.isna(raw) else None
            )
            if pdt <= now_utc:
                now_idx = len(depth_dates) - 1
    resort["snow_depth_dates"] = depth_dates
    resort["snow_depth_values"] = depth_values
    resort["snow_depth_now_idx"] = now_idx

    # ── Daily: snowfall, precip, temp_max, temp_min, weather_code ────────
    daily = response.Daily()
    daily_snow_cm = daily.Variables(0).ValuesAsNumpy()
    # Variables(1) = precipitation_sum  (not stored per-resort, skip)
    daily_temp_max = daily.Variables(2).ValuesAsNumpy()
    daily_temp_min = daily.Variables(3).ValuesAsNumpy()
    daily_dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )

    # Indices for the next FORECAST_DAYS days starting today (UTC)
    forecast_idx = [
        j for j, dt in enumerate(daily_dates)
        if today_utc <= dt.date() < today_utc + timedelta(days=FORECAST_DAYS)
    ]

    # Next N-day snowfall sum
    next_snow = sum(_safe_float(daily_snow_cm[j]) for j in forecast_idx)
    resort["next_10_snow_cm"] = round(next_snow, 1)

    # Recent 7-day snowfall (from past 7 days up to yesterday)
    recent_idx = [
        j for j, dt in enumerate(daily_dates)
        if today_utc - timedelta(days=7) <= dt.date() < today_utc
    ]
    recent_snow = sum(_safe_float(daily_snow_cm[j]) for j in recent_idx)
    resort["recent_7day_snow_cm"] = round(recent_snow, 1)

    # Min / avg / max temperature
    all_temps = []
    for j in forecast_idx:
        if not pd.isna(daily_temp_min[j]):
            all_temps.append(float(daily_temp_min[j]))
        if not pd.isna(daily_temp_max[j]):
            all_temps.append(float(daily_temp_max[j]))
    resort["min_10day_temp"] = round(min(all_temps), 1) if all_temps else None
    resort["max_10day_temp"] = round(max(all_temps), 1) if all_temps else None
    resort["avg_10day_temp"] = round(sum(all_temps) / len(all_temps), 1) if all_temps else None

    # Daily series for chart (next FORECAST_DAYS days)
    resort["forecast_dates"] = [daily_dates[j].strftime("%b %d") for j in forecast_idx]
    resort["forecast_daily_snow"] = [
        round(_safe_float(daily_snow_cm[j]), 1) for j in forecast_idx
    ]


def _fetch_forecast_batch(resorts: list, indices: list, model: str, client):
    """Fetch forecast for one batch of resorts sharing the same model."""
    group = [resorts[i] for i in indices]
    params = {
        "latitude": [r["lat"] for r in group],
        "longitude": [r["lon"] for r in group],
        "hourly": "snow_depth",
        "daily": [
            "snowfall_sum",
            "precipitation_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "weather_code",
        ],
        "timezone": [r["timezone"] for r in group],
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "models": model,
    }

    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()

    try:
        responses = client.weather_api(FORECAST_URL, params=params)
    except Exception as exc:
        print(f"  [forecast] ERROR fetching model={model}: {exc}")
        for i in indices:
            _set_forecast_defaults(resorts[i])
        return

    for j, response in enumerate(responses):
        resort = resorts[indices[j]]
        try:
            _process_forecast_response(resort, response, now_utc, today_utc)
        except Exception as exc:
            print(f"  [forecast] ERROR processing {resort['name']}: {exc}")
            _set_forecast_defaults(resort)


def fetch_forecast_data(resorts: list, client):
    """
    Group resorts by their ``model`` field and issue one batch forecast
    request per model.  Resorts that do not specify a model default to
    ``best_match``.
    """
    model_groups: dict[str, list[int]] = defaultdict(list)
    for i, resort in enumerate(resorts):
        model_groups[resort.get("model", "best_match")].append(i)

    for model, indices in model_groups.items():
        names = [resorts[i]["name"] for i in indices]
        print(f"  [forecast] model={model}  resorts={names}")
        _fetch_forecast_batch(resorts, indices, model, client)


# ─────────────────────────────────────────────────────────────────────────────
# Archive API (season-to-date snowfall)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_archive_group(resorts: list, indices: list, season_start: date, end_date: date, client):
    if not indices:
        return
    if season_start > end_date:
        for i in indices:
            resorts[i]["season_total_cm"] = 0.0
        return

    group = [resorts[i] for i in indices]
    params = {
        "latitude": [r["lat"] for r in group],
        "longitude": [r["lon"] for r in group],
        "start_date": season_start.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "snowfall_sum",
    }

    try:
        responses = client.weather_api(ARCHIVE_URL, params=params)
    except Exception as exc:
        print(f"  [archive] ERROR fetching {season_start} – {end_date}: {exc}")
        for i in indices:
            resorts[i].setdefault("season_total_cm", 0.0)
        return

    for j, response in enumerate(responses):
        resort = resorts[indices[j]]
        try:
            daily = response.Daily()
            snow = daily.Variables(0).ValuesAsNumpy()
            total = float(sum(_safe_float(v) for v in snow))
            resort["season_total_cm"] = round(total, 1)
        except Exception as exc:
            print(f"  [archive] ERROR processing {resort['name']}: {exc}")
            resort["season_total_cm"] = 0.0


def fetch_archive_data(resorts: list, client):
    """
    Fetch season-to-date snowfall (ERA5 reanalysis via archive API).
    NH and SH resorts are batched separately because their ski seasons
    start at different times of year.
    """
    today = date.today()
    end_date = today - timedelta(days=1)  # archive lags by ~1–5 days

    nh_indices = [i for i, r in enumerate(resorts) if r["lat"] >= -10]
    sh_indices = [i for i, r in enumerate(resorts) if r["lat"] < -10]

    nh_season_start = get_season_start_date(45.0, today)   # northern hemisphere
    sh_season_start = get_season_start_date(-45.0, today)  # southern hemisphere

    print(f"  [archive] NH season start: {nh_season_start}  ({len(nh_indices)} resorts)")
    _fetch_archive_group(resorts, nh_indices, nh_season_start, end_date, client)

    print(f"  [archive] SH season start: {sh_season_start}  ({len(sh_indices)} resorts)")
    _fetch_archive_group(resorts, sh_indices, sh_season_start, end_date, client)


# ─────────────────────────────────────────────────────────────────────────────
# Ski Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_ski_scores(resorts: list):
    """
    Compute a 0–100 Ski Score for every resort.

    Weights:
      30 % – Current snow base (150 cm → 100 pts)
      40 % – Next-10-day snowfall forecast (60 cm → 100 pts)
      20 % – Season-to-date snowfall total (400 cm → 100 pts)
      10 % – Cold-temperature bonus (≤ −15 °C min → 100 pts)
    """
    for resort in resorts:
        snow_base = float(resort.get("snow_base_cm") or 0)
        next_snow = float(resort.get("next_10_snow_cm") or 0)
        season = float(resort.get("season_total_cm") or 0)
        min_temp = resort.get("min_10day_temp")

        base_score = min(snow_base / 150.0, 1.0) * 100
        forecast_score = min(next_snow / 60.0, 1.0) * 100
        season_score = min(season / 400.0, 1.0) * 100
        cold_score = (
            max(0.0, min((-min_temp) / 15.0, 1.0)) * 100
            if min_temp is not None else 0.0
        )

        resort["ski_score"] = round(
            base_score * 0.30
            + forecast_score * 0.40
            + season_score * 0.20
            + cold_score * 0.10
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_data(resorts: list, cache_file: str):
    """Fetch forecast + archive data and compute ski scores for all resorts."""
    client = _make_client(cache_file)

    print("[1/3] Fetching forecast data …")
    fetch_forecast_data(resorts, client)

    print("[2/3] Fetching archive data (season totals) …")
    fetch_archive_data(resorts, client)

    print("[3/3] Computing ski scores …")
    compute_ski_scores(resorts)
