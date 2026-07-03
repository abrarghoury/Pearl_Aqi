"""
app/components/page_trend.py
Page 2 — AQI Trend: actual history + model predictions + future forecast
"""

import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta


ISTANBUL_IMAGE_URL = "https://images.pexels.com/photos/36520727/pexels-photo-36520727.jpeg"
CHART_BG           = "#1c2128"
CHART_GRID         = "#30363d"


def _clean_html(html: str) -> str:
    return re.sub(r"(?m)^[ \t]+", "", html)


def _white_heading(text: str, level: str = "h3") -> str:
    return _clean_html(
        f"<{level} style='color:#ffffff; font-weight:700; margin-bottom:8px;'>{text}</{level}>"
    )


def _inject_page_css():
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 12px 14px;
        }
        [data-testid="stMetricLabel"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }
        [data-testid="stMetricDelta"] {
            font-weight: 600 !important;
        }
        .chart-card {
            background: #1c2128;
            border: 1px solid #30363d;
            border-radius: 14px;
            padding: 10px 6px 4px 6px;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_trend_chart(
    history: pd.DataFrame,
    past_preds: pd.DataFrame,
    prediction: dict,
) -> go.Figure:

    fig = go.Figure()

    if not history.empty and "aqi_mean" in history.columns:
        fig.add_trace(go.Scatter(
            x    = history["date"],
            y    = history["aqi_mean"],
            mode = "lines+markers",
            name = "Actual AQI",
            line = dict(color="#4fc3f7", width=2.5),
            marker = dict(size=5, color="#4fc3f7"),
            hovertemplate = "<b>%{x|%d %b}</b><br>Actual AQI: %{y:.0f}<extra></extra>",
        ))

    if not past_preds.empty and "day1" in past_preds.columns:
        past_preds = past_preds.copy()
        try:
            past_preds["pred_aqi"] = past_preds["day1"].apply(
                lambda x: x.get("aqi") if isinstance(x, dict) else None
            )
            past_preds = past_preds.dropna(subset=["pred_aqi"])

            fig.add_trace(go.Scatter(
                x    = past_preds["prediction_date"],
                y    = past_preds["pred_aqi"],
                mode = "lines+markers",
                name = "Model Prediction (Day1)",
                line = dict(color="#ff9800", width=2, dash="dot"),
                marker = dict(size=5, symbol="diamond", color="#ff9800"),
                hovertemplate = "<b>%{x|%d %b}</b><br>Predicted AQI: %{y:.0f}<extra></extra>",
            ))
        except Exception:
            pass

    if not history.empty and not past_preds.empty and "pred_aqi" in past_preds.columns:
        merged = pd.merge(
            history[["date", "aqi_mean"]],
            past_preds[["prediction_date", "pred_aqi"]],
            left_on="date", right_on="prediction_date",
            how="inner",
        )
        big_errors = merged[abs(merged["aqi_mean"] - merged["pred_aqi"]) > 20]

        if not big_errors.empty:
            fig.add_trace(go.Scatter(
                x    = big_errors["date"],
                y    = big_errors["aqi_mean"],
                mode = "markers",
                name = "Large Error (>20 AQI)",
                marker = dict(size=14, color="#ff1744", symbol="x",
                              line=dict(width=2, color="#ffffff")),
                hovertemplate = (
                    "<b>%{x|%d %b}</b><br>Actual: %{y:.0f}<br>Error flagged<extra></extra>"
                ),
            ))

    pred_date   = prediction["prediction_date"]
    latest_aqi  = prediction["latest_aqi"]
    latest_date = prediction["latest_date"]

    future_dates = [
        latest_date,
        pred_date,
        pred_date + timedelta(days=1),
        pred_date + timedelta(days=2),
    ]
    future_aqi = [
        latest_aqi,
        prediction["day1"]["aqi"],
        prediction["day2"]["aqi"],
        prediction["day3"]["aqi"],
    ]

    fig.add_trace(go.Scatter(
        x    = future_dates,
        y    = future_aqi,
        mode = "lines+markers+text",
        name = "Forecast",
        line = dict(color="#ce93d8", width=2.5, dash="dash"),
        marker = dict(size=10, color="#ce93d8", symbol="circle"),
        text = [f"{int(v)}" for v in future_aqi],
        textposition = "top center",
        textfont = dict(color="#ce93d8", size=12),
        hovertemplate = "<b>%{x|%d %b}</b><br>Forecast AQI: %{y:.0f}<extra></extra>",
    ))

    day2_aqi    = prediction["day2"]["aqi"]
    day3_aqi    = prediction["day3"]["aqi"]
    shade_dates = [pred_date + timedelta(days=1), pred_date + timedelta(days=2)]
    upper       = [day2_aqi * 1.15, day3_aqi * 1.25]
    lower       = [day2_aqi * 0.85, day3_aqi * 0.75]

    fig.add_trace(go.Scatter(
        x         = shade_dates + shade_dates[::-1],
        y         = upper + lower[::-1],
        fill      = "toself",
        fillcolor = "rgba(206, 147, 216, 0.12)",
        line      = dict(color="rgba(0,0,0,0)"),
        name      = "Forecast Uncertainty",
        hoverinfo = "skip",
    ))

    fig.add_hrect(
        y0=150, y1=500,
        fillcolor = "rgba(255, 0, 0, 0.07)",
        line_width = 0,
        annotation_text = "Unhealthy Zone (>150)",
        annotation_position = "top left",
        annotation_font = dict(color="#ff6b6b", size=11),
    )
    fig.add_hline(
        y=150,
        line_dash="dot",
        line_color="#ff6b6b",
        line_width=1.5,
        opacity=0.6,
    )

    fig.update_layout(
        title       = dict(text="AQI Trend — Last 30 Days + 3-Day Forecast",
                           font=dict(size=18, color="#ffffff"), x=0),
        paper_bgcolor = CHART_BG,
        plot_bgcolor  = CHART_BG,
        legend = dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color="#e6e6e6", size=12), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis = dict(showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
                     tickformat="%d %b", tickfont=dict(color="#c9d1d9"), zeroline=False),
        yaxis = dict(showgrid=True, gridcolor=CHART_GRID, gridwidth=1,
                     tickfont=dict(color="#c9d1d9"),
                     title=dict(text="AQI", font=dict(color="#c9d1d9")),
                     rangemode="tozero", zeroline=False),
        hovermode = "x unified",
        margin    = dict(l=10, r=10, t=60, b=10),
        height    = 480,
    )
    return fig


def _build_error_table(history: pd.DataFrame, past_preds: pd.DataFrame) -> pd.DataFrame:
    if history.empty or past_preds.empty:
        return pd.DataFrame()

    try:
        preds = past_preds.copy()
        preds["pred_aqi"] = preds["day1"].apply(
            lambda x: x.get("aqi") if isinstance(x, dict) else None
        )
        merged = pd.merge(
            history[["date", "aqi_mean"]],
            preds[["prediction_date", "pred_aqi"]],
            left_on="date", right_on="prediction_date", how="inner",
        ).dropna(subset=["pred_aqi"])

        merged["error"]     = (merged["pred_aqi"] - merged["aqi_mean"]).round(1)
        merged["abs_error"] = merged["error"].abs().round(1)
        merged["accuracy"]  = (
            (1 - merged["abs_error"] / merged["aqi_mean"].replace(0, 1)).clip(0, 1) * 100
        ).round(1)

        display = merged[["date", "aqi_mean", "pred_aqi", "error", "accuracy"]].copy()
        display.columns = ["Date", "Actual AQI", "Predicted AQI", "Error", "Accuracy %"]
        display["Date"] = display["Date"].dt.strftime("%d %b %Y")
        return display.sort_values("Date", ascending=False).head(14)

    except Exception:
        return pd.DataFrame()


def _pollutant_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if history.empty:
        return fig

    if "pm2_5_mean" in history.columns:
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["pm2_5_mean"],
            mode="lines", name="PM2.5",
            line=dict(color="#ef9a9a", width=2),
            hovertemplate="<b>%{x|%d %b}</b><br>PM2.5: %{y:.1f}<extra></extra>",
        ))

    if "pm10_mean" in history.columns:
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["pm10_mean"],
            mode="lines", name="PM10",
            line=dict(color="#ffcc80", width=2),
            hovertemplate="<b>%{x|%d %b}</b><br>PM10: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title         = dict(text="Pollutant Trend — PM2.5 & PM10",
                             font=dict(color="#ffffff", size=15)),
        paper_bgcolor = CHART_BG,
        plot_bgcolor  = CHART_BG,
        xaxis  = dict(showgrid=True, gridcolor=CHART_GRID,
                      tickformat="%d %b", tickfont=dict(color="#c9d1d9")),
        yaxis  = dict(showgrid=True, gridcolor=CHART_GRID,
                      tickfont=dict(color="#c9d1d9"),
                      title=dict(text="µg/m³", font=dict(color="#c9d1d9"))),
        legend = dict(font=dict(color="#e6e6e6"), bgcolor="rgba(0,0,0,0)"),
        hovermode = "x unified",
        margin    = dict(l=10, r=10, t=50, b=10),
        height    = 280,
    )
    return fig


# ---------------------------------------------------------------------------
# Istanbul Forecast Summary — 50/50 image + metric tiles (pure HTML, no st.metric)
# ---------------------------------------------------------------------------

def _summary_section(prediction: dict):
    """
    Both columns rendered as a single HTML block so heights are controlled
    by CSS and not by Streamlit's own column layout engine.
    Using st.metric inside st.columns caused the image column to be shorter
    than the metrics column because st.metric adds invisible padding/spacing
    that Streamlit does not expose. Pure HTML gives pixel-perfect control.
    """
    current_aqi = int(prediction["latest_aqi"])
    day1_aqi    = int(prediction["day1"]["aqi"])
    day2_aqi    = int(prediction["day2"]["aqi"])
    day3_aqi    = int(prediction["day3"]["aqi"])

    cat_cur = prediction["day1"]["category"]
    cat_d1  = prediction["day1"]["category"]
    cat_d2  = prediction["day2"]["category"]
    cat_d3  = prediction["day3"]["category"]

    pred_date = prediction["prediction_date"]

    def _fmt(d):
        return f"{d.day} {d.strftime('%b %Y')}"

    def _delta_html(val: int, ref: int) -> str:
        diff = val - ref
        if diff > 0:
            return (f"<span style='color:#ff6b6b; font-size:13px; font-weight:700;'>"
                    f"&#x2191; +{diff}</span>")
        elif diff < 0:
            return (f"<span style='color:#56d364; font-size:13px; font-weight:700;'>"
                    f"&#x2193; {diff}</span>")
        else:
            return "<span style='color:#8b949e; font-size:13px;'>&#x2212; 0</span>"

    def _tile(title: str, val: int, date_str: str, color: str, delta_html: str) -> str:
        return f"""
<div style="
    background:#1c2128;
    border:1px solid #30363d;
    border-left:3px solid {color};
    border-radius:10px;
    padding:16px 18px;
    flex:1;
">
    <div style="font-size:11px; color:#8b949e; font-weight:600;
                letter-spacing:0.8px; text-transform:uppercase; margin-bottom:6px;">
        {title}
    </div>
    <div style="font-size:36px; font-weight:900; color:#ffffff; line-height:1; margin-bottom:4px;">
        {val}
    </div>
    {delta_html}
    <div style="font-size:10px; color:#484f58; margin-top:6px;">{date_str}</div>
    <div style="background:#21262d; border-radius:3px; height:3px; margin-top:10px;">
        <div style="width:{min(100,(val/500)*100):.1f}%;
                    background:{color}; height:3px; border-radius:3px;"></div>
    </div>
</div>
"""

    tile_current = _tile("Current AQI",    current_aqi, "Today",             cat_cur["bg_color"], "")
    tile_d1      = _tile("Day 1 Forecast", day1_aqi,    _fmt(pred_date),     cat_d1["bg_color"],  _delta_html(day1_aqi, current_aqi))
    tile_d2      = _tile("Day 2 Forecast", day2_aqi,    _fmt(pred_date + timedelta(days=1)), cat_d2["bg_color"], _delta_html(day2_aqi, day1_aqi))
    tile_d3      = _tile("Day 3 Forecast", day3_aqi,    _fmt(pred_date + timedelta(days=2)), cat_d3["bg_color"], _delta_html(day3_aqi, day2_aqi))

    html = f"""
<div style="display:flex; gap:16px; align-items:stretch; min-height:340px;">

    <!-- LEFT — Istanbul image, fills exact same height as right col -->
    <div style="flex:1; border-radius:14px; overflow:hidden;
                border:1px solid #30363d; min-height:340px;">
        <img src="{ISTANBUL_IMAGE_URL}"
             style="width:100%; height:100%; object-fit:cover; display:block;">
    </div>

    <!-- RIGHT — 2x2 metric tiles grid -->
    <div style="flex:1; display:flex; flex-direction:column; gap:12px;">
        <!-- Row 1 -->
        <div style="display:flex; gap:12px; flex:1;">
            {tile_current}
            {tile_d1}
        </div>
        <!-- Row 2 -->
        <div style="display:flex; gap:12px; flex:1;">
            {tile_d2}
            {tile_d3}
        </div>
    </div>

</div>
"""
    st.markdown(_clean_html(html), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(prediction: dict):
    _inject_page_css()

    st.markdown(_white_heading("AQI Trend", "h1"), unsafe_allow_html=True)

    history    = prediction.get("feature_history", pd.DataFrame())
    past_preds = prediction.get("past_predictions", pd.DataFrame())

    if history.empty:
        st.warning("No historical data available. Run the feature pipeline first.")
        return

    # Main trend chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(_build_trend_chart(history, past_preds, prediction),
                    use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Istanbul Forecast Summary — 50/50
    st.markdown("---")
    st.markdown(_white_heading("Istanbul Forecast Summary"), unsafe_allow_html=True)
    _summary_section(prediction)

    # Pollutant trend
    st.markdown("---")
    st.markdown(_white_heading("Pollutant Breakdown"), unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(_pollutant_chart(history), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Accuracy table
    st.markdown("---")
    st.markdown(_white_heading("Model Accuracy — Last 14 Days"), unsafe_allow_html=True)

    error_df = _build_error_table(history, past_preds)

    if error_df.empty:
        st.info("Not enough prediction history yet. Accuracy table builds up after a few daily runs.")
    else:
        def _color_error(val):
            try:
                v = float(val)
                if abs(v) > 30:  return "color:#ff6b6b; font-weight:600"
                if abs(v) > 15:  return "color:#ffb74d"
                return "color:#81c784"
            except Exception:
                return ""

        avg_acc = error_df["Accuracy %"].mean()

        a1, a2 = st.columns(2)
        with a1:
            st.metric("Average Accuracy", f"{avg_acc:.1f}%")
        with a2:
            if "abs_error" in error_df.columns:
                st.metric("Average Absolute Error", f"{error_df['abs_error'].mean():.1f} AQI")

        st.dataframe(
            error_df.drop(columns=["abs_error"], errors="ignore")
                    .style.applymap(_color_error, subset=["Error"]),
            use_container_width=True,
            hide_index=True,
        )