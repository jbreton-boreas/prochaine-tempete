# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
ou-skier.py – Global Ski Destination Advisor

Fetches snow conditions and 10-day forecasts for ~54 worldwide ski resorts
using Open-Meteo, ranks them by Ski Score, and generates a static HTML page.
"""

import os
import sys
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader
from highcharts_core.chart import Chart

# Resolve paths relative to this script (works regardless of cwd)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, "templates")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CACHE_FILE = os.path.join(SCRIPT_DIR, ".cache")

sys.path.insert(0, SCRIPT_DIR)  # ensure local modules are importable
from resorts import resorts       # noqa: E402
from fetch_data import fetch_all_data  # noqa: E402

templates_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Number of top resorts to include in the charts
TOP_N_CHART = 10


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_series(resort_list: list, data_key: str, multiplier: float = 1.0) -> list:
    return [
        {
            "name": r["name"],
            "data": [
                round(v * multiplier, 1) if v is not None else None
                for v in r.get(data_key, [])
            ],
        }
        for r in resort_list
    ]


def plot_forecast_chart(top_resorts: list) -> str:
    """
    Line chart: daily snowfall forecast (cm) for the next FORECAST_DAYS days,
    one series per top resort.
    """
    if not top_resorts:
        return ""
    try:
        categories = top_resorts[0].get("forecast_dates", [])
        series = _build_series(top_resorts, "forecast_daily_snow")

        chart = Chart(
            container="forecast_chart",
            options={
                "chart": {
                    "type": "column",
                    "height": 450,
                },
                "title": {"text": "10-Day Daily Snowfall Forecast (cm) – Top Resorts"},
                "xAxis": {"categories": categories},
                "yAxis": {"title": {"text": "Snowfall (cm)"}},
                "series": series,
                "plotOptions": {
                    "column": {"grouping": True, "pointPadding": 0.05, "groupPadding": 0.1}
                },
                "tooltip": {"shared": True, "valueSuffix": " cm"},
                "legend": {"enabled": True},
            },
        )
        return chart.to_js_literal()
    except Exception as exc:
        print(f"Error in plot_forecast_chart: {exc}")
        return ""


def plot_snow_depth_chart(top_resorts: list) -> str:
    """
    Line chart: hourly snow depth (cm) for past 7 days + next 10 days,
    one series per top resort.  A red vertical line marks 'now'.
    """
    if not top_resorts:
        return ""
    try:
        # Use the longest date array as x-axis categories
        categories = max(
            (r.get("snow_depth_dates", []) for r in top_resorts),
            key=len,
            default=[],
        )
        now_idx = top_resorts[0].get("snow_depth_now_idx", 0)
        series = _build_series(top_resorts, "snow_depth_values")

        chart = Chart(
            container="snow_depth_chart",
            options={
                "chart": {"type": "line", "height": 450},
                "title": {"text": "Snow Base Depth (cm) – Past 7 Days & 10-Day Forecast"},
                "xAxis": {
                    "categories": categories,
                    "tickInterval": 24,  # show one label per day (hourly data)
                    "plotLines": [
                        {
                            "color": "#FF0000",
                            "width": 2,
                            "value": now_idx,
                            "label": {"text": "Now", "style": {"color": "#FF0000"}},
                        }
                    ],
                },
                "yAxis": {"title": {"text": "Snow Depth (cm)"}, "min": 0},
                "series": series,
                "tooltip": {"shared": False, "valueSuffix": " cm"},
                "legend": {"enabled": True},
            },
        )
        return chart.to_js_literal()
    except Exception as exc:
        print(f"Error in plot_snow_depth_chart: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_html(
    resort_list: list,
    forecast_chart_js: str,
    snow_depth_chart_js: str,
) -> str:
    now_utc = datetime.now(timezone.utc)
    data = {
        "resorts": resort_list,
        "last_update": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
    }

    template = templates_env.get_template("ou_skier.html")
    output = template.render(
        data=data,
        forecast_chart_js=forecast_chart_js,
        snow_depth_chart_js=snow_depth_chart_js,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"HTML written to {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    try:
        fetch_all_data(resorts, CACHE_FILE)

        # Sort by ski score descending
        resorts.sort(key=lambda r: r.get("ski_score", 0), reverse=True)

        # Top-N resorts for charts (skip resorts with no data)
        top_resorts = [r for r in resorts if r.get("ski_score", 0) > 0][:TOP_N_CHART]
        if not top_resorts:
            top_resorts = resorts[:TOP_N_CHART]

        forecast_chart_js = plot_forecast_chart(top_resorts)
        snow_depth_chart_js = plot_snow_depth_chart(top_resorts)

        generate_html(resorts, forecast_chart_js, snow_depth_chart_js)
        print("Done.")

    except Exception as exc:
        exc_type, _, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"Fatal error: {exc} [{exc_type}] in {fname}:{exc_tb.tb_lineno}")
        sys.exit(1)


if __name__ == "__main__":
    main()
