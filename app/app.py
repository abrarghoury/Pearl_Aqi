"""
app/app.py
Main Streamlit entry point.
Run: streamlit run app/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import streamlit as st
from datetime import timedelta

from app.components import page_prediction, page_trend, page_shap, page_model_metrics
from app.predict import run_prediction
from config.settings import settings

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title            = f"AQI Forecast — {settings.CITY_NAME}",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Base page/sidebar backgrounds now come from .streamlit/config.toml
       (backgroundColor + secondaryBackgroundColor) — that's the reliable
       way to set them since Streamlit's theme engine owns those surfaces.
       Only card-level, hover, and font polish lives here. */

    .block-container { padding-top:1.5rem; padding-bottom:2rem; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] div {
        color: #ffffff !important;
    }

    /* All headings white */
    h1, h2, h3, h4 { color: #ffffff !important; }

    /* ----------------------------------------------------------------
       Navigation buttons — resting-state look only. Colors on click/
       focus/hover now come from primaryColor in config.toml, so there's
       nothing left here fighting Streamlit's own theme engine.
    ---------------------------------------------------------------- */
    div[data-testid="stSidebar"] button[kind="primary"] {
        font-weight: 700 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background: #1c2128 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #262c36 !important;
        border-color: #8b949e !important;
        color: #ffffff !important;
    }

    /* Kill the mobile browser's default tap-highlight overlay, which can
       also look like a stray red/grey flash on click. */
    button, a { -webkit-tap-highlight-color: transparent; }

    .js-plotly-plot { border-radius: 12px; }

    details {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 4px 8px;
    }

    ::-webkit-scrollbar { width:6px; height:6px; }
    ::-webkit-scrollbar-track { background:#0b0e14; }
    ::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }
    ::-webkit-scrollbar-thumb:hover { background:#555; }

    .stDataFrame { border-radius:10px; overflow:hidden; }
    hr { border-color:#30363d; margin:1.5rem 0; }
    .stCaption { color:#8b949e !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(prediction: dict) -> str:
    with st.sidebar:

        # Logo / title
        st.markdown(
            f"""
            <div style="text-align:center; padding:16px 0 20px 0;">
                <div style="font-size:19px; font-weight:800; color:#ffffff; margin-top:8px;">
                    AQI Forecast
                </div>
                <div style="font-size:12px; color:#8b949e; margin-top:3px;">
                    {settings.CITY_NAME}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<hr style='border-color:#262c36; margin:0 0 16px 0;'>",
            unsafe_allow_html=True,
        )

        # Current AQI card
        current_aqi = int(prediction["latest_aqi"])
        category    = prediction["day1"]["category"]
        latest_date = prediction["latest_date"].strftime("%d %b %Y")

        st.markdown(
            f"""
            <div style="
                background:{category['bg_color']}18;
                border:1px solid {category['bg_color']}50;
                border-radius:12px;
                padding:16px;
                text-align:center;
                margin-bottom:16px;
            ">
                <div style="color:#8b949e; font-size:10px; font-weight:600;
                            letter-spacing:1px; margin-bottom:6px;">
                    CURRENT AQI · {latest_date}
                </div>
                <div style="font-size:52px; font-weight:900;
                            color:white;line-height:1; margin-bottom:8px;">
                    {current_aqi}
                </div>
                <div style="
                    display:inline-block;
                    background:{category['bg_color']};
                    color:#ffffff !important;
                    padding:5px 16px;
                    border-radius:20px;
                    font-size:12px;
                    font-weight:700;
                ">
                    {category['label']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3-day forecast rows
        st.markdown(
            "<div style='color:#8b949e; font-size:10px; font-weight:700; "
            "letter-spacing:1.5px; margin-bottom:8px;'>3-DAY FORECAST</div>",
            unsafe_allow_html=True,
        )

        pred_date = prediction["prediction_date"]
        day_dates = [
            pred_date,
            pred_date + timedelta(days=1),
            pred_date + timedelta(days=2),
        ]

        for key, date_obj in zip(["day1", "day2", "day3"], day_dates):
            d   = prediction[key]
            cat = d["category"]
            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    padding:9px 12px;
                    margin-bottom:6px;
                    background:#1c2128;
                    border-radius:8px;
                    border-left:3px solid {cat['bg_color']};
                ">
                    <span style="color:#c9d1d9; font-size:12px; font-weight:500;">
                        {date_obj.strftime('%d %b')}
                    </span>
                    <span style="display:flex; align-items:center; gap:6px;">
                        <span style="color:{cat['bg_color']}; font-weight:800; font-size:15px;">
                            {int(d['aqi'])}
                        </span>
                        <span style="font-size:10px; color:#ffffff; font-weight:600;">
                            {cat['label']}
                        </span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "<hr style='border-color:#262c36; margin:14px 0;'>",
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown(
            "<div style='color:#8b949e; font-size:10px; font-weight:700; "
            "letter-spacing:1.5px; margin-bottom:10px;'>NAVIGATION</div>",
            unsafe_allow_html=True,
        )

        pages = {
            "Predictive AQI Forecast":        "prediction",
            "Historical Trends & Patterns":   "trend",
            "Explainable AI (SHAP Insights)": "shap",
            "Model Performance & Evaluation": "metrics",
        }

        if "active_page" not in st.session_state:
            st.session_state.active_page = "prediction"

        for page_label, page_key in pages.items():
            is_active = st.session_state.active_page == page_key
            if st.button(
                page_label,
                key                 = f"nav_{page_key}",
                use_container_width = True,
                type                = "primary" if is_active else "secondary",
            ):
                st.session_state.active_page = page_key
                st.rerun()

        st.markdown(
            "<hr style='border-color:#262c36; margin:14px 0;'>",
            unsafe_allow_html=True,
        )

        # Active models
        metadata = prediction.get("model_metadata", [])
        if metadata:
            st.markdown(
                "<div style='color:#8b949e; font-size:10px; font-weight:700; "
                "letter-spacing:1.5px; margin-bottom:8px;'>ACTIVE MODELS</div>",
                unsafe_allow_html=True,
            )
            label_map = {
                settings.TARGET_DAY1: "Day 1",
                settings.TARGET_DAY2: "Day 2",
                settings.TARGET_DAY3: "Day 3",
            }
            for m in metadata:
                target     = m.get("target", "")
                model_name = m.get("model_name", "—")
                version    = m.get("version",    "—")
                r2         = m.get("test_r2",    "—")
                day_label  = label_map.get(target, target)

                st.markdown(
                    f"""
                    <div style="
                        padding:8px 10px;
                        margin-bottom:5px;
                        background:#1c2128;
                        border-radius:8px;
                        border:1px solid #262c36;
                    ">
                        <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                            <span style="color:#ffffff; font-size:12px; font-weight:600;">
                                {day_label}
                            </span>
                            <span style="color:#8b949e; font-size:10px;">{version}</span>
                        </div>
                        <div style="color:#8b949e; font-size:11px;">
                            {model_name}
                            <span style="color:#4fc3f7; font-weight:600;"> R² {r2}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<hr style='border-color:#262c36; margin:14px 0;'>",
            unsafe_allow_html=True,
        )

        # Refresh button
        if st.button("Refresh Data", use_container_width=True, type="secondary"):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            """
            <div style="text-align:center; color:#3d444d; font-size:10px;
                        margin-top:16px; line-height:1.8;">
                Data: Open-Meteo API<br>
                Store: MongoDB Atlas<br>
                Models retrain daily
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.active_page


# ---------------------------------------------------------------------------
# Error screen
# ---------------------------------------------------------------------------

def _render_error(msg: str):
    st.markdown(
        f"""
        <div style="
            background:#2d1b1b;
            border:1px solid #8b3a3a;
            border-radius:12px;
            padding:40px;
            text-align:center;
            margin-top:60px;
        ">
            <div style="font-size:36px; margin-bottom:16px;">⚠️</div>
            <div style="color:#d97a7a; font-size:18px; font-weight:700; margin-bottom:12px;">
                Pipeline Error
            </div>
            <div style="color:#c9d1d9; font-size:14px; line-height:1.7;">{msg}</div>
            <div style="color:#8b949e; font-size:12px; margin-top:20px;">
                Run feature pipeline → training pipeline → refresh.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with st.spinner("Loading predictions..."):
        try:
            prediction = run_prediction()
        except RuntimeError as e:
            _render_error(str(e))
            logger.error("Prediction failed: %s", e, exc_info=True)
            st.stop()
        except Exception as e:
            _render_error(f"Unexpected error: {e}")
            logger.error("Unexpected error: %s", e, exc_info=True)
            st.stop()

    active_page = _render_sidebar(prediction)

    if active_page == "prediction":
        page_prediction.render(prediction)
    elif active_page == "trend":
        page_trend.render(prediction)
    elif active_page == "shap":
        page_shap.render(prediction)
    elif active_page == "metrics":
        page_model_metrics.render(prediction)
    else:
        page_prediction.render(prediction)


if __name__ == "__main__":
    main()