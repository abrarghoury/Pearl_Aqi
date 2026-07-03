"""
app/components/page_prediction.py
Page 1 — 3-Day AQI Forecast Dashboard (Day 1, 2, 3 cards + alerts)
"""

import re
import streamlit as st
from datetime import timedelta
from app.predict import get_aqi_category


def _clean_html(html: str) -> str:
    """
    Strip leading whitespace from every line of an HTML string.

    Streamlit's markdown parser treats any line indented by 4+ spaces as a
    code block, which causes raw HTML/comments to be printed as plain text
    instead of being rendered. Every HTML string passed to st.markdown()
    must go through this before rendering.
    """
    return re.sub(r"(?m)^[ \t]+", "", html)


def _inject_font():
    # Loads a clean, modern dashboard font (Inter) and applies it app-wide.
    # NOTE: ideally this <style> block should live once in app.py so it applies
    # to all 4 pages consistently — for now it's injected here since we're
    # polishing this page first.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero_banner_html(prediction: dict) -> str:
    # Big, prominent hero banner — Istanbul skyline as the visual anchor of the page
    current_aqi = int(prediction["latest_aqi"])
    category    = prediction["day1"]["category"]
    latest_date = prediction["latest_date"].strftime("%d %b %Y")

    banner_image_url = "https://images.pexels.com/photos/37292055/pexels-photo-37292055.jpeg"

    html = f"""
    <div style="
        position: relative;
        border-radius: 16px;
        overflow: hidden;
        padding: 40px 44px;
        margin-bottom: 22px;
        background-image:
            linear-gradient(90deg, rgba(8,10,14,0.90) 0%, rgba(8,10,14,0.55) 50%, rgba(8,10,14,0.15) 100%),
            url('{banner_image_url}');
        background-size: cover;
        background-position: center 55%;
        height: 270px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size:12px; color:#e8b923; font-weight:700; letter-spacing:1.5px; margin-bottom:10px;">
            📍 ISTANBUL, TURKEY
        </div>
        <div style="font-size:36px; font-weight:800; color:#ffffff; margin-bottom:6px; line-height:1.2;">
            3-Day Air Quality Forecast
        </div>
        <div style="font-size:13px; color:#c9d1d9; margin-bottom:18px;">
            Last updated: {latest_date}
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:15px; color:#ffffff; font-weight:600;">
                Current AQI: {current_aqi}
            </span>
            <span style="display:inline-flex; align-items:center; gap:7px;">
                <span style="
                    width:9px; height:9px; border-radius:50%;
                    background:{category['bg_color']};
                    display:inline-block;
                    box-shadow: 0 0 6px {category['bg_color']}aa;
                "></span>
                <span style="color:#ffffff; font-weight:700; font-size:14px;">
                    {category['label']}
                </span>
            </span>
        </div>
    </div>
    """
    return _clean_html(html)


def _aqi_gauge_html(aqi_value: float, category: dict) -> str:
    # Compact gauge bar — maps AQI 0-500 to 0-100% width, clamped
    pct = min(100, max(0, (aqi_value / 500) * 100))

    html = f"""
    <div style="margin-top:4px;">
        <div style="background:#30363d; border-radius:6px; height:5px; width:100%;">
            <div style="
                width:{pct:.1f}%;
                background:{category['bg_color']};
                height:5px;
                border-radius:6px;
                transition: width 0.5s ease;
            "></div>
        </div>
        <div style="display:flex; justify-content:space-between;
                    font-size:9px; color:#6e7681; margin-top:2px;">
            <span>0</span><span>100</span><span>200</span>
            <span>300</span><span>500</span>
        </div>
    </div>
    """
    return _clean_html(html)


def _prediction_card(
    relative_label: str,
    full_date_label: str,
    aqi_value: float,
    category: dict,
    confidence: str,
) -> str:
    # Compact card with a muted, category-tinted background — colorful but
    # dark enough to stay easy on the eyes against the app's dark theme.
    html = f"""
    <div style="
        background: linear-gradient(135deg, {category['bg_color']}14, {category['bg_color']}26);
        border: 1px solid {category['bg_color']}40;
        border-left: 3px solid {category['bg_color']};
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
        height: 100%;
    ">
        <div style="font-size:10px; color:#c9d1d9; font-weight:600; margin-bottom:1px; letter-spacing:0.8px; text-transform:uppercase;">
            {relative_label}
        </div>
        <div style="font-size:13px; color:#ffffff; font-weight:700; margin-bottom:8px;">
            {full_date_label}
        </div>

        <!-- AQI number — always white for readability, regardless of category -->
        <div style="
            font-size:34px;
            font-weight:800;
            color:#ffffff;
            line-height:1;
            margin-bottom:6px;
            text-shadow: 0 2px 8px {category['bg_color']}55;
        ">
            {int(round(aqi_value))}
        </div>

        <!-- Category badge -->
        <div style="
            display:inline-block;
            background:{category['bg_color']};
            color:{category['text_color']};
            padding:3px 12px;
            border-radius:16px;
            font-size:11px;
            font-weight:700;
            margin-bottom:7px;
            letter-spacing:0.3px;
        ">
            {category['label']}
        </div>

        <!-- Confidence tag -->
        <div style="font-size:10px; color:#c9d1d9; margin-top:1px;">
            Confidence: <b style="color:#ffffff;">{confidence}</b>
        </div>
    </div>
    """
    return _clean_html(html)


def _health_advisory(label: str) -> dict:
    # What people should actually do based on AQI category
    advisories = {
        "Good": {
            "icon": "✅",
            "text": "Air quality is satisfactory. Enjoy outdoor activities.",
            "color": "#006400",
        },
        "Moderate": {
            "icon": "🟡",
            "text": "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion.",
            "color": "#ffd60a",
        },
        "Unhealthy (Sensitive)": {
            "icon": "🟠",
            "text": "Sensitive groups (elderly, children, asthma) should reduce outdoor activity.",
            "color": "#dc7611",
        },
        "Unhealthy": {
            "icon": "🔴",
            "text": "Everyone may begin to experience health effects. Limit outdoor exertion.",
            "color": "#de0404",
        },
        "Very Unhealthy": {
            "icon": "🟣",
            "text": "Health alert — everyone should avoid prolonged outdoor activity.",
            "color": "#8f3d97",
        },
        "Hazardous": {
            "icon": "🚨",
            "text": "Health emergency. Avoid all outdoor activity. Wear N95 if must go out.",
            "color": "#7a0425",
        },
    }
    return advisories.get(label, advisories["Moderate"])


def _advisory_box_html(advisory: dict, worst_label: str) -> str:
    # Custom advisory banner — replaces st.success/warning/error so the
    # message text is always white and readable against the dark theme.
    color = advisory["color"]
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}1a, {color}30);
        border-left: 4px solid {color};
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 14px;
        color: #ffffff;
        line-height: 1.5;
    ">
        <span style="font-size:16px;">{advisory['icon']}</span>
        <b style="color:#ffffff;">{worst_label}</b> — {advisory['text']}
    </div>
    """
    return _clean_html(html)


def render(prediction: dict):
    """
    Main render function called by app.py.
    prediction dict comes directly from predict.run_prediction().
    """
    _inject_font()

    st.markdown(_hero_banner_html(prediction), unsafe_allow_html=True)

    # Pull confidence levels from model metadata
    confidence_map = {
        m["target"]: m.get("confidence", "—")
        for m in prediction.get("model_metadata", [])
    }

    pred_date = prediction["prediction_date"]

    # Full, human-readable date shown on each card, e.g. "1 July 2026"
    def _full_date(d):
        day = str(d.day)  # no leading zero, e.g. "1" not "01"
        return f"{day} {d.strftime('%B %Y')}"

    day_configs = [
        {
            "key":             "day1",
            "relative_label":  "Day 1",
            "full_date_label": _full_date(pred_date),
            "confidence":      confidence_map.get("aqi_day1", "High"),
        },
        {
            "key":             "day2",
            "relative_label":  "Day 2",
            "full_date_label": _full_date(pred_date + timedelta(days=1)),
            "confidence":      confidence_map.get("aqi_day2", "Moderate"),
        },
        {
            "key":             "day3",
            "relative_label":  "Day 3",
            "full_date_label": _full_date(pred_date + timedelta(days=2)),
            "confidence":      confidence_map.get("aqi_day3", "Low"),
        },
    ]

    # Render 3 cards side by side
    cols = st.columns(3, gap="medium")

    for col, cfg in zip(cols, day_configs):
        data     = prediction[cfg["key"]]
        aqi_val  = data["aqi"]
        category = data["category"]

        with col:
            st.markdown(
                _prediction_card(
                    cfg["relative_label"],
                    cfg["full_date_label"],
                    aqi_val,
                    category,
                    cfg["confidence"],
                ),
                unsafe_allow_html=True,
            )
            # Gauge bar below each card
            st.markdown(
                _aqi_gauge_html(aqi_val, category),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    # Worst-case category across 3 days drives the alert banner
    all_labels = [prediction[k]["category"]["label"] for k in ["day1", "day2", "day3"]]
    severity_order = [
        "Good", "Moderate", "Unhealthy (Sensitive)",
        "Unhealthy", "Very Unhealthy", "Hazardous",
    ]
    worst_label = max(all_labels, key=lambda l: severity_order.index(l))
    advisory    = _health_advisory(worst_label)

    st.markdown("---")
    st.markdown("### 🏥 Health Advisory")
    st.markdown(_advisory_box_html(advisory, worst_label), unsafe_allow_html=True)

    # AQI reference scale at the bottom
    st.markdown("---")
    st.markdown("### 📊 AQI Scale Reference")

    scale_cols = st.columns(6)
    scale_items = [
        ("0–50",    "Good",                 "#05be05", "#000"),
        ("51–100",  "Moderate",             "#e5e50f", "#000"),
        ("101–150", "Unhealthy (Sen.)",     "#f07a05", "#fff"),
        ("151–200", "Unhealthy",            "#ed0303", "#fff"),
        ("201–300", "Very Unhealthy",       "#802b88", "#fff"),
        ("301–500", "Hazardous",            "#730222", "#fff"),
    ]

    for col, (rng, label, bg, fg) in zip(scale_cols, scale_items):
        with col:
            st.markdown(
                _clean_html(
                    f"""
                    <div style="
                        background:{bg}; color:{fg};
                        border-radius:10px; padding:8px 6px;
                        text-align:center; font-size:11px;
                        font-weight:600;
                    ">
                        <div style="font-size:13px; font-weight:800;">{rng}</div>
                        {label}
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )