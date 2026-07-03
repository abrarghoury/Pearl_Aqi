"""
app/components/page_shap.py
Page 3 — SHAP Explainability: bar chart, waterfall, day comparison, insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.predict import (
    get_top_shap_features,
    build_waterfall_data,
    generate_shap_insight,
)


# ---------------------------------------------------------------------------
# Shared styling (hardcoded — independent of Streamlit's light/dark theme)
# ---------------------------------------------------------------------------

CARD_BG   = "#1c1f26"
CARD_BG_2 = "#20242d"
TEXT_MAIN = "#f5f5f5"
TEXT_SUB  = "#a8adb8"
ACCENT_1  = "#4fc3f7"
ACCENT_2  = "#7e57c2"


def _inject_base_css():
    st.markdown(
        f"""
        <style>
        .shap-table {{
            width: 100%;
            border-collapse: collapse;
            border-radius: 12px;
            overflow: hidden;
            font-size: 14px;
        }}
        .shap-table th {{
            text-align: left;
            padding: 12px 14px;
            color: {TEXT_MAIN};
            font-weight: 700;
            background: linear-gradient(90deg, {ACCENT_1} 0%, {ACCENT_2} 100%);
        }}
        .shap-table td {{
            padding: 10px 14px;
            color: {TEXT_MAIN};
            border-bottom: 1px solid #2c313c;
        }}
        .shap-table tr:nth-child(even) td {{
            background: {CARD_BG_2};
        }}
        .shap-table tr:nth-child(odd) td {{
            background: {CARD_BG};
        }}

        /* ---- Expander header (e.g. "Raw SHAP Values Table") ---- */
        [data-testid="stExpander"] summary {{
            background: {CARD_BG_2};
            border-radius: 8px;
        }}
        [data-testid="stExpander"] summary p {{
            color: {TEXT_MAIN} !important;
            font-weight: 700 !important;
            font-size: 15px !important;
        }}
        [data-testid="stExpander"] summary svg {{
            fill: {TEXT_MAIN} !important;
        }}

        /* ---- Tabs (Day 1 / Day 2 / Day 3) ---- */
        button[data-baseweb="tab"] {{
            color: {TEXT_SUB} !important;
        }}
        button[data-baseweb="tab"] p {{
            color: {TEXT_SUB} !important;
            font-weight: 600 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {TEXT_MAIN} !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: {TEXT_MAIN} !important;
        }}
        [data-baseweb="tab-highlight"] {{
            background-color: {ACCENT_1} !important;
        }}
        [data-baseweb="tab-border"] {{
            background-color: #2c313c !important;
        }}
        [data-baseweb="tab-list"] {{
            gap: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _shap_bar_chart(shap_df: pd.DataFrame, day_label: str, color: str) -> go.Figure:
    """
    Horizontal bar chart — top N features sorted by absolute SHAP value.
    Positive = pushed AQI up (red), Negative = pulled AQI down (green).
    """
    if shap_df.empty:
        return go.Figure()

    # Sort ascending so highest bar appears at top in horizontal layout
    df = shap_df.sort_values("abs_shap", ascending=True).tail(10)

    bar_colors = [
        "#e57373" if v > 0 else "#81c784"
        for v in df["shap_value"]
    ]

    fig = go.Figure(go.Bar(
        x           = df["shap_value"],
        y           = df["feature"],
        orientation = "h",
        marker_color = bar_colors,
        text        = [f"{v:+.2f}" for v in df["shap_value"]],
        textposition = "outside",
        textfont    = dict(color=TEXT_SUB, size=11),
        hovertemplate = (
            "<b>%{y}</b><br>"
            "SHAP value: %{x:+.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color="#555555", line_width=1.5)

    fig.update_layout(
        title       = dict(text=f"Feature Impact — {day_label}", font=dict(color=TEXT_MAIN, size=14)),
        paper_bgcolor = CARD_BG,
        plot_bgcolor  = CARD_BG,
        xaxis = dict(
            showgrid  = True,
            gridcolor = "#2a2a2a",
            tickfont  = dict(color=TEXT_SUB),
            title     = dict(text="SHAP Value (AQI impact)", font=dict(color=TEXT_SUB, size=11)),
            zeroline  = False,
        ),
        yaxis = dict(
            tickfont  = dict(color=TEXT_MAIN, size=11),
            showgrid  = False,
        ),
        margin = dict(l=10, r=60, t=50, b=10),
        height = 380,
    )

    return fig


def _waterfall_chart(waterfall_df: pd.DataFrame, base_value: float, final_pred: float, day_label: str) -> go.Figure:
    """
    Waterfall chart — shows step-by-step AQI buildup from base value
    (average model output) to the final prediction.
    Each bar = one feature's contribution.
    """
    if waterfall_df is None or waterfall_df.empty:
        return go.Figure()

    features  = list(waterfall_df["feature"])
    shap_vals = list(waterfall_df["shap_value"])
    colors    = list(waterfall_df["color"])

    # Add base value bar at start and final prediction at end
    all_x      = ["Base Value"] + features + ["Final Prediction"]
    all_values = [base_value] + shap_vals + [0]  # 0 = total bar handled by measure
    all_colors = ["#546e7a"] + colors + ["#7e57c2"]

    # Plotly waterfall uses measure to distinguish relative vs total bars
    measures = ["absolute"] + ["relative"] * len(features) + ["total"]

    fig = go.Figure(go.Waterfall(
        x             = all_x,
        y             = all_values,
        measure       = measures,
        text          = [f"{v:+.1f}" if m == "relative" else f"{v:.1f}"
                         for v, m in zip(all_values, measures)],
        textposition  = "outside",
        textfont      = dict(color=TEXT_SUB, size=10),
        connector     = dict(line=dict(color="#444444", width=1, dash="dot")),
        increasing    = dict(marker=dict(color="#e57373")),
        decreasing    = dict(marker=dict(color="#81c784")),
        totals        = dict(marker=dict(color="#7e57c2")),
        hovertemplate = "<b>%{x}</b><br>%{y:+.2f} AQI<extra></extra>",
    ))

    fig.update_layout(
        title       = dict(text=f"Prediction Breakdown — {day_label}", font=dict(color=TEXT_MAIN, size=14)),
        paper_bgcolor = CARD_BG,
        plot_bgcolor  = CARD_BG,
        xaxis = dict(
            tickfont  = dict(color=TEXT_MAIN, size=10),
            showgrid  = False,
            tickangle = -30,
        ),
        yaxis = dict(
            showgrid  = True,
            gridcolor = "#2a2a2a",
            tickfont  = dict(color=TEXT_SUB),
            title     = dict(text="AQI", font=dict(color=TEXT_SUB, size=11)),
        ),
        showlegend = False,
        margin     = dict(l=10, r=20, t=50, b=60),
        height     = 400,
    )

    return fig


def _day_comparison_chart(prediction: dict) -> go.Figure:
    """
    Side-by-side grouped bar chart comparing top-5 feature SHAP values
    across Day1, Day2, Day3 — shows how feature importance shifts
    as the forecast horizon extends.
    """
    day_keys = ["day1", "day2", "day3"]
    day_labels = ["Day 1", "Day 2", "Day 3"]
    colors     = ["#4fc3f7", "#ce93d8", "#ffcc80"]

    # Collect top features per day
    all_top_features = set()
    day_shap_maps = {}

    for key in day_keys:
        data = prediction[key]
        sv   = data.get("shap_values")
        fn   = data.get("shap_features", [])

        if sv is None:
            day_shap_maps[key] = {}
            continue

        top_df = get_top_shap_features(sv, fn, top_n=5)
        fmap   = dict(zip(top_df["feature"], top_df["shap_value"]))
        day_shap_maps[key] = fmap
        all_top_features.update(fmap.keys())

    if not all_top_features:
        return go.Figure()

    # Union of all top features — fill missing with 0
    feature_list = sorted(all_top_features)

    fig = go.Figure()

    for key, label, color in zip(day_keys, day_labels, colors):
        fmap   = day_shap_maps.get(key, {})
        values = [fmap.get(f, 0.0) for f in feature_list]

        fig.add_trace(go.Bar(
            name  = label,
            x     = feature_list,
            y     = values,
            marker_color = color,
            text  = [f"{v:+.1f}" for v in values],
            textposition = "outside",
            textfont = dict(size=9, color=TEXT_SUB),
            hovertemplate = f"<b>{label}</b><br>%{{x}}<br>SHAP: %{{y:+.2f}}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#555555", line_width=1)

    fig.update_layout(
        title       = dict(text="Feature Importance Shift — Day 1 vs Day 2 vs Day 3", font=dict(color=TEXT_MAIN, size=14)),
        paper_bgcolor = CARD_BG,
        plot_bgcolor  = CARD_BG,
        barmode     = "group",
        bargroupgap = 0.15,
        xaxis = dict(
            tickfont  = dict(color=TEXT_MAIN, size=10),
            tickangle = -25,
            showgrid  = False,
        ),
        yaxis = dict(
            showgrid  = True,
            gridcolor = "#2a2a2a",
            tickfont  = dict(color=TEXT_SUB),
            title     = dict(text="SHAP Value", font=dict(color=TEXT_SUB, size=11)),
            zeroline  = False,
        ),
        legend = dict(
            orientation = "h",
            y           = 1.12,
            x           = 0,
            font        = dict(color=TEXT_MAIN),
            bgcolor     = "rgba(0,0,0,0)",
        ),
        margin = dict(l=10, r=10, t=70, b=60),
        height = 400,
    )

    return fig


# ---------------------------------------------------------------------------
# Insight text block
# ---------------------------------------------------------------------------

def _render_insight_card(text: str, day_label: str, aqi_val: float, bg_color: str):
    st.markdown(
        f"""
        <div style="
            background: {CARD_BG};
            border-left: 4px solid {bg_color};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
        ">
            <div style="color:{TEXT_SUB}; font-size:12px; margin-bottom:4px;">
                {day_label} — Predicted AQI <b style="color:{TEXT_MAIN};">{int(round(aqi_val))}</b>
            </div>
            <div style="color:{TEXT_MAIN}; font-size:14px; line-height:1.6;">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Raw SHAP table — proper HTML table (replaces st.dataframe styling)
# ---------------------------------------------------------------------------

def _render_shap_table(top_df: pd.DataFrame):
    display = top_df[["feature", "shap_value"]].copy()
    display.columns = ["Feature", "SHAP Value"]
    display["SHAP Value"] = display["SHAP Value"].round(4)
    display["Direction"]  = display["SHAP Value"].apply(
        lambda v: "Increases AQI" if v > 0 else "Decreases AQI"
    )

    body_rows = ""
    for _, row in display.iterrows():
        dir_color = "#e57373" if row["Direction"] == "Increases AQI" else "#81c784"
        body_rows += (
            "<tr>"
            f"<td>{row['Feature']}</td>"
            f"<td>{row['SHAP Value']:+.4f}</td>"
            f"<td style='color:{dir_color}; font-weight:600;'>{row['Direction']}</td>"
            "</tr>"
        )

    table_html = f"""
    <table class="shap-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>SHAP Value</th>
                <th>Direction</th>
            </tr>
        </thead>
        <tbody>
            {body_rows}
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(prediction: dict):
    _inject_base_css()

    st.title("SHAP Explainability")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows how much each input feature "
        "pushed the model's AQI prediction up or down from the average baseline."
    )

    day_configs = [
        {
            "key":       "day1",
            "label":     "Day 1",
            "color":     "#4fc3f7",
        },
        {
            "key":       "day2",
            "label":     "Day 2",
            "color":     "#ce93d8",
        },
        {
            "key":       "day3",
            "label":     "Day 3",
            "color":     "#ffcc80",
        },
    ]

    # ------------------------------------------------------------------
    # Section 1 — Plain English Insights
    # ------------------------------------------------------------------
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700; margin-top:10px;'>"
        f"What's Driving the Forecast</div>",
        unsafe_allow_html=True,
    )

    any_shap = any(
        prediction[cfg["key"]].get("shap_values") is not None
        for cfg in day_configs
    )

    if not any_shap:
        st.warning(
            "SHAP values are not available — the winning model may not support "
            "the current explainer configuration. Check logs for details."
        )
        return

    for cfg in day_configs:
        data       = prediction[cfg["key"]]
        shap_vals  = data.get("shap_values")
        feat_names = data.get("shap_features", [])
        aqi_val    = data["aqi"]

        if shap_vals is None:
            continue

        insight = generate_shap_insight(shap_vals, feat_names, day_label=cfg["label"])
        _render_insight_card(insight, cfg["label"], aqi_val, cfg["color"])

    # ------------------------------------------------------------------
    # Section 2 — SHAP Bar Charts (one per day, in tabs)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700;'>"
        f"Feature Impact — Top 10 Per Day</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Day 1", "Day 2", "Day 3"])
    tabs = [tab1, tab2, tab3]

    for tab, cfg in zip(tabs, day_configs):
        with tab:
            data       = prediction[cfg["key"]]
            shap_vals  = data.get("shap_values")
            feat_names = data.get("shap_features", [])

            if shap_vals is None:
                st.info(f"SHAP not available for {cfg['label']}.")
                continue

            top_df = get_top_shap_features(shap_vals, feat_names, top_n=10)

            if top_df.empty:
                st.info("No SHAP features to display.")
                continue

            bar_fig = _shap_bar_chart(top_df, cfg["label"], cfg["color"])
            st.plotly_chart(bar_fig, use_container_width=True)

            # Most impactful feature callout
            top_row   = top_df.iloc[0]
            direction = "increased" if top_row["shap_value"] > 0 else "decreased"
            st.markdown(
                f"<div style='color:{TEXT_SUB}; font-size:13px;'>"
                f"<b style='color:{TEXT_MAIN};'>Most impactful:</b> "
                f"<code>{top_row['feature']}</code> {direction} the forecast by "
                f"<b style='color:{TEXT_MAIN};'>{abs(top_row['shap_value']):.1f} AQI points</b>.</div>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Section 3 — Waterfall Charts
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700;'>"
        f"Prediction Breakdown — Base to Final</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{TEXT_SUB}; font-size:13px; margin-bottom:10px;'>"
        f"Each bar shows one feature's contribution. "
        f"Starting from the model's average output (base value) to final prediction.</div>",
        unsafe_allow_html=True,
    )

    w1, w2, w3 = st.columns(3)
    wcols = [w1, w2, w3]

    for col, cfg in zip(wcols, day_configs):
        with col:
            data       = prediction[cfg["key"]]
            shap_vals  = data.get("shap_values")
            feat_names = data.get("shap_features", [])
            base_val   = data.get("base_value")
            aqi_val    = data["aqi"]

            if shap_vals is None or base_val is None:
                st.info(f"Waterfall not available — {cfg['label']}")
                continue

            wf_df = build_waterfall_data(shap_vals, feat_names, base_val, top_n=8)
            wf_fig = _waterfall_chart(wf_df, base_val, aqi_val, cfg["label"])
            st.plotly_chart(wf_fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Section 4 — Cross-Day Comparison
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700;'>"
        f"Feature Importance Shift Across Forecast Horizon</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{TEXT_SUB}; font-size:13px; margin-bottom:10px;'>"
        f"Shows how the same features gain or lose influence as prediction horizon "
        f"grows from Day 1 to Day 3. Forecast weather features (fc_*) typically "
        f"dominate Day 2 and Day 3.</div>",
        unsafe_allow_html=True,
    )

    comparison_fig = _day_comparison_chart(prediction)

    if comparison_fig.data:
        st.plotly_chart(comparison_fig, use_container_width=True)
    else:
        st.info("Comparison chart needs SHAP values for at least one day.")

    # ------------------------------------------------------------------
    # Section 5 — Raw SHAP values table (collapsible, horizontal day tabs)
    # ------------------------------------------------------------------
    with st.expander("Raw SHAP Values Table"):
        raw_tab1, raw_tab2, raw_tab3 = st.tabs(["Day 1", "Day 2", "Day 3"])
        raw_tabs = [raw_tab1, raw_tab2, raw_tab3]

        for raw_tab, cfg in zip(raw_tabs, day_configs):
            with raw_tab:
                data       = prediction[cfg["key"]]
                shap_vals  = data.get("shap_values")
                feat_names = data.get("shap_features", [])

                if shap_vals is None:
                    st.info(f"SHAP not available for {cfg['label']}.")
                    continue

                top_df = get_top_shap_features(shap_vals, feat_names, top_n=20)

                if top_df.empty:
                    st.info("No SHAP features to display.")
                    continue

                _render_shap_table(top_df)