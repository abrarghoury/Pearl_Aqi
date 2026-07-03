"""
app/components/page_model_metrics.py
Page 4 — Model Performance: R², RMSE, MAE, scatter, residuals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pymongo import MongoClient

from config.settings import settings


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _get_db():
    client = MongoClient(settings.MONGODB_URI)
    return client[settings.MONGODB_DB_NAME]


def _load_all_predictions_vs_actuals() -> dict:
    db = _get_db()

    actuals = list(
        db[settings.COLLECTION_FEATURES].find(
            {"aqi_mean": {"$exists": True}},
            {"_id": 0, "date": 1, "aqi_mean": 1}
        )
    )
    preds = list(db[settings.COLLECTION_PREDICTIONS].find({}, {"_id": 0}))

    empty = {"day1": pd.DataFrame(), "day2": pd.DataFrame(), "day3": pd.DataFrame()}

    if not actuals or not preds:
        return empty

    actuals_df = pd.DataFrame(actuals)
    actuals_df["date"] = pd.to_datetime(actuals_df["date"])

    preds_df = pd.DataFrame(preds)
    preds_df["prediction_date"] = pd.to_datetime(preds_df["prediction_date"])

    result = {}

    for i, key in enumerate(["day1", "day2", "day3"], start=1):
        if key not in preds_df.columns:
            result[key] = pd.DataFrame()
            continue

        try:
            pred_col = preds_df[["prediction_date", key]].copy()
            pred_col["pred_aqi"] = pred_col[key].apply(
                lambda x: x.get("aqi") if isinstance(x, dict) else None
            )
            pred_col["target_date"] = pred_col["prediction_date"] + pd.Timedelta(days=i - 1)

            merged = pd.merge(
                actuals_df.rename(columns={"aqi_mean": "actual"}),
                pred_col[["target_date", "pred_aqi"]].rename(
                    columns={"target_date": "date", "pred_aqi": "predicted"}
                ),
                on="date", how="inner"
            ).dropna(subset=["actual", "predicted"])

            result[key] = merged.sort_values("date").reset_index(drop=True)

        except Exception:
            result[key] = pd.DataFrame()

    return result


def _compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 2:
        return {"rmse": None, "mae": None, "r2": None, "n": 0}

    actual    = df["actual"].values
    predicted = df["predicted"].values
    residuals = actual - predicted

    rmse   = float(np.sqrt(np.mean(residuals ** 2)))
    mae    = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4), "n": len(df)}


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------

CARD_BG   = "#1c1f26"   # fixed dark card background (independent of app theme)
CARD_BG_2 = "#20242d"   # slightly lighter alt row
TEXT_MAIN = "#f5f5f5"   # near-white, always readable on CARD_BG
TEXT_SUB  = "#a8adb8"   # muted gray-blue for labels

DAY_COLORS = {"day1": "#4fc3f7", "day2": "#ce93d8", "day3": "#ffcc80"}


def _inject_base_css():
    st.markdown(
        f"""
        <style>
        .mm-section-title {{
            color: {TEXT_MAIN};
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 2px;
        }}
        .mm-caption {{
            color: {TEXT_SUB};
            font-size: 13px;
            margin-bottom: 14px;
        }}
        .mm-table {{
            width: 100%;
            border-collapse: collapse;
            border-radius: 12px;
            overflow: hidden;
            font-size: 14px;
        }}
        .mm-table th {{
            text-align: left;
            padding: 12px 14px;
            color: #0d1117;
            font-weight: 700;
        }}
        .mm-table td {{
            padding: 10px 14px;
            color: {TEXT_MAIN};
            border-bottom: 1px solid #2c313c;
        }}
        .mm-table tr:nth-child(even) td {{
            background: {CARD_BG_2};
        }}
        .mm-table tr:nth-child(odd) td {{
            background: {CARD_BG};
        }}
        .mm-table td:first-child {{
            color: {TEXT_SUB};
            font-weight: 600;
        }}
        .mm-live-card {{
            background: {CARD_BG};
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            border: 1px solid #2c313c;
        }}
        .mm-live-title {{
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        .mm-live-r2 {{
            font-size: 30px;
            font-weight: 900;
            margin-bottom: 6px;
        }}
        .mm-live-sub {{
            color: {TEXT_MAIN};
            font-size: 13px;
        }}
        .mm-live-n {{
            color: {TEXT_SUB};
            font-size: 11px;
            margin-top: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _scatter_chart(df: pd.DataFrame, day_label: str, color: str) -> go.Figure:
    fig = go.Figure()

    if df.empty:
        return fig

    actual    = df["actual"].values
    predicted = df["predicted"].values
    n         = len(df)
    opacity   = np.linspace(0.3, 1.0, n).tolist()

    fig.add_trace(go.Scatter(
        x          = actual,
        y          = predicted,
        mode       = "markers",
        name       = "Predictions",
        marker     = dict(size=8, color=color, opacity=opacity, line=dict(color="#fff", width=0.5)),
        customdata = df["date"].dt.strftime("%d %b %Y").values,
        hovertemplate = (
            "<b>%{customdata}</b><br>"
            "Actual: %{x:.0f}<br>"
            "Predicted: %{y:.0f}<extra></extra>"
        ),
    ))

    axis_min = min(actual.min(), predicted.min()) * 0.9
    axis_max = max(actual.max(), predicted.max()) * 1.1

    fig.add_trace(go.Scatter(
        x=[ axis_min, axis_max], y=[axis_min, axis_max],
        mode="lines", name="Perfect Prediction",
        line=dict(color="#ffffff", width=1.5, dash="dash"),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max], y=[axis_min + 20, axis_max + 20],
        mode="lines", line=dict(color="#555", width=1, dash="dot"),
        name="±20 AQI Band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[axis_min, axis_max], y=[axis_min - 20, axis_max - 20],
        mode="lines", line=dict(color="#555", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(255,255,255,0.04)",
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        title         = dict(text=f"Actual vs Predicted — {day_label}", font=dict(color="#fff", size=14)),
        paper_bgcolor = CARD_BG,
        plot_bgcolor  = CARD_BG,
        xaxis = dict(title=dict(text="Actual AQI", font=dict(color="#aaa", size=11)),
                     showgrid=True, gridcolor="#2a2a2a", tickfont=dict(color="#aaa"), zeroline=False),
        yaxis = dict(title=dict(text="Predicted AQI", font=dict(color="#aaa", size=11)),
                     showgrid=True, gridcolor="#2a2a2a", tickfont=dict(color="#aaa"), zeroline=False),
        legend = dict(font=dict(color="#ccc", size=11), bgcolor="rgba(0,0,0,0)"),
        margin = dict(l=10, r=10, t=50, b=10),
        height = 380,
    )
    return fig


def _residual_chart(df: pd.DataFrame, day_label: str) -> go.Figure:
    if df.empty:
        return go.Figure()

    residuals = df["actual"].values - df["predicted"].values
    fig = go.Figure()

    fig.add_hline(y=0, line_color="#ffffff", line_width=1.5, line_dash="dash", opacity=0.4)
    fig.add_hrect(
        y0=-20, y1=20,
        fillcolor="rgba(255,255,255,0.04)", line_width=0,
        annotation_text="±20 AQI", annotation_position="top right",
        annotation_font=dict(color="#888", size=10),
    )

    fig.add_trace(go.Bar(
        x=df["date"], y=residuals,
        name="Residual",
        marker_color=["#e57373" if r > 0 else "#81c784" for r in residuals],
        hovertemplate="<b>%{x|%d %b}</b><br>Residual: %{y:+.1f}<extra></extra>",
    ))

    fig.update_layout(
        title         = dict(text=f"Residuals — {day_label}", font=dict(color="#fff", size=14)),
        paper_bgcolor = CARD_BG,
        plot_bgcolor  = CARD_BG,
        xaxis = dict(showgrid=True, gridcolor="#2a2a2a", tickformat="%d %b", tickfont=dict(color="#aaa")),
        yaxis = dict(showgrid=True, gridcolor="#2a2a2a", tickfont=dict(color="#aaa"),
                     title=dict(text="Actual − Predicted", font=dict(color="#aaa", size=11)),
                     zeroline=False),
        showlegend = False,
        margin     = dict(l=10, r=10, t=50, b=10),
        height     = 280,
    )
    return fig


def _rolling_rmse_chart(data: dict) -> go.Figure:
    fig    = go.Figure()
    labels = {"day1": "Day 1", "day2": "Day 2", "day3": "Day 3"}

    for key, color in DAY_COLORS.items():
        df = data.get(key, pd.DataFrame())
        if df.empty or len(df) < 7:
            continue

        df = df.copy()
        df["residual_sq"]   = (df["actual"] - df["predicted"]) ** 2
        df["rolling_rmse"]  = (
            df["residual_sq"].rolling(window=7, min_periods=3).mean().apply(np.sqrt)
        )

        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rolling_rmse"],
            mode="lines", name=labels[key],
            line=dict(color=color, width=2),
            hovertemplate=(
                f"<b>{labels[key]}</b><br>%{{x|%d %b}}<br>"
                "Rolling RMSE: %{y:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title         = dict(text="Rolling 7-Day RMSE — All Horizons", font=dict(color="#fff", size=14)),
        paper_bgcolor = CARD_BG,
        plot_bgcolor  = CARD_BG,
        xaxis  = dict(showgrid=True, gridcolor="#2a2a2a", tickformat="%d %b", tickfont=dict(color="#aaa")),
        yaxis  = dict(showgrid=True, gridcolor="#2a2a2a", tickfont=dict(color="#aaa"),
                      title=dict(text="RMSE (AQI)", font=dict(color="#aaa", size=11)), rangemode="tozero"),
        legend = dict(font=dict(color="#ccc"), bgcolor="rgba(0,0,0,0)"),
        hovermode = "x unified",
        margin    = dict(l=10, r=10, t=50, b=10),
        height    = 300,
    )
    return fig


# ---------------------------------------------------------------------------
# Model evaluation table (replaces broken st.metric cards)
# ---------------------------------------------------------------------------

def _render_training_table(meta_by_key: dict, day_configs: list):
    rows = [
        ("Model",     "model_name"),
        ("Version",   "version"),
        ("CV RMSE",   "cv_rmse"),
        ("Test RMSE", "test_rmse"),
        ("Test MAE",  "test_mae"),
        ("Test R²",   "test_r2"),
        ("Features",  "feature_count"),
        ("Rows Used", "data_rows_used"),
        ("Trained At", "trained_at"),
    ]

    header_cells = ""
    for cfg in day_configs:
        header_cells += (
            f'<th style="background:{cfg["color"]};">{cfg["label"]}</th>'
        )

    body_rows = ""
    for label, field in rows:
        row_cells = f"<td>{label}</td>"
        for cfg in day_configs:
            m = meta_by_key.get(cfg["key"])
            if m is None:
                val = "—"
            else:
                val = m.get(field, "—")
                if field == "trained_at" and val not in (None, "—"):
                    val = val.strftime("%d %b %Y %H:%M")
                if val is None:
                    val = "—"
            row_cells += f"<td>{val}</td>"
        body_rows += f"<tr>{row_cells}</tr>"

    table_html = f"""
    <table class="mm-table">
        <thead>
            <tr>
                <th style="background:#3a3f4b; color:{TEXT_MAIN};">Metric</th>
                {header_cells}
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

    st.markdown(
        f"<div class='mm-section-title'>Model Performance</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mm-caption'>Training metrics from the model registry, "
        "plus live accuracy once predictions accumulate over time.</div>",
        unsafe_allow_html=True,
    )

    day_configs = [
        {"key": "day1", "label": "Day 1", "color": "#4fc3f7"},
        {"key": "day2", "label": "Day 2", "color": "#ce93d8"},
        {"key": "day3", "label": "Day 3", "color": "#ffcc80"},
    ]

    # ------------------------------------------------------------------
    # Training metrics — proper table, always-readable colors
    # ------------------------------------------------------------------
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700; "
        f"margin-top:10px;'>Training Metrics (Model Registry)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='mm-caption'>Computed on the held-out 20% test set during training.</div>",
        unsafe_allow_html=True,
    )

    metadata    = prediction.get("model_metadata", [])
    meta_by_key = {}
    for m in metadata:
        t = m.get("target", "")
        if t == settings.TARGET_DAY1:
            meta_by_key["day1"] = m
        elif t == settings.TARGET_DAY2:
            meta_by_key["day2"] = m
        elif t == settings.TARGET_DAY3:
            meta_by_key["day3"] = m

    if not meta_by_key:
        st.info("No trained models found in the registry yet.")
    else:
        _render_training_table(meta_by_key, day_configs)

    # ------------------------------------------------------------------
    # Live accuracy
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700;'>"
        f"Live Accuracy (Predictions vs Actuals)</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mm-caption'>Fills in after a few daily runs as predictions "
        "accumulate in MongoDB.</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading prediction history..."):
        live_data = _load_all_predictions_vs_actuals()

    live_available = any(
        not live_data.get(cfg["key"], pd.DataFrame()).empty
        for cfg in day_configs
    )

    if not live_available:
        st.info(
            "No live accuracy data yet — this section fills in after "
            "the dashboard has been running for a few days."
        )
    else:
        live_cols = st.columns(3, gap="medium")

        for col, cfg in zip(live_cols, day_configs):
            with col:
                df      = live_data.get(cfg["key"], pd.DataFrame())
                metrics = _compute_metrics(df)
                color   = cfg["color"]

                r2_str   = f"{metrics['r2']:.4f}"  if metrics["r2"]   is not None else "N/A"
                rmse_str = f"{metrics['rmse']:.2f}" if metrics["rmse"] is not None else "N/A"
                mae_str  = f"{metrics['mae']:.2f}"  if metrics["mae"]  is not None else "N/A"

                st.markdown(
                    f"""
                    <div class="mm-live-card">
                        <div class="mm-live-title" style="color:{color};">
                            {cfg['label']} — Live
                        </div>
                        <div class="mm-live-r2" style="color:{color};">R² {r2_str}</div>
                        <div class="mm-live-sub">RMSE {rmse_str}</div>
                        <div class="mm-live-sub">MAE {mae_str}</div>
                        <div class="mm-live-n">{metrics['n']} predictions evaluated</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------------
    # Scatter plots
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700;'>"
        f"Actual vs Predicted Scatter</div>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Day 1", "Day 2", "Day 3"])

    for tab, cfg in zip([tab1, tab2, tab3], day_configs):
        with tab:
            df = live_data.get(cfg["key"], pd.DataFrame())
            if df.empty:
                st.info("Not enough live data yet.")
                continue
            st.plotly_chart(_scatter_chart(df, cfg["label"], cfg["color"]), use_container_width=True)
            st.plotly_chart(_residual_chart(df, cfg["label"]), use_container_width=True)

    # ------------------------------------------------------------------
    # Rolling RMSE
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='color:{TEXT_MAIN}; font-size:18px; font-weight:700;'>"
        f"Model Stability Over Time</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='mm-caption'>Rising RMSE = model drifting. Flat = model holding up well.</div>",
        unsafe_allow_html=True,
    )

    rolling_fig = _rolling_rmse_chart(live_data)
    if rolling_fig.data:
        st.plotly_chart(rolling_fig, use_container_width=True)
    else:
        st.info("Needs at least 7 days of prediction history.")

    # ------------------------------------------------------------------
    # Full log
    # ------------------------------------------------------------------
    with st.expander("Full Prediction Log"):
        for cfg in day_configs:
            df = live_data.get(cfg["key"], pd.DataFrame())
            if df.empty:
                continue

            st.markdown(f"**{cfg['label']}**")
            display = df[["date", "actual", "predicted"]].copy()
            display["error"]    = (display["predicted"] - display["actual"]).round(1)
            display["accuracy"] = (
                (1 - display["error"].abs() / display["actual"].replace(0, 1))
                .clip(0, 1) * 100
            ).round(1)
            display["date"] = display["date"].dt.strftime("%d %b %Y")
            display.columns = ["Date", "Actual", "Predicted", "Error", "Accuracy %"]

            def _color_err(val):
                try:
                    v = float(val)
                    if abs(v) > 30:  return "color:#ff6b6b; font-weight:600"
                    if abs(v) > 15:  return "color:#ffb74d"
                    return "color:#81c784"
                except Exception:
                    return ""

            st.dataframe(
                display.sort_values("Date", ascending=False)
                       .style.applymap(_color_err, subset=["Error"]),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)