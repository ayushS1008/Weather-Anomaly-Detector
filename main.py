from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Weather Anomaly Detector",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { --green:#267a4a; --ink:#17221f; --muted:#687872; --line:#dce5e0; }
    .stApp { background:#f3f6f4; color:var(--ink); }
    .block-container { max-width:1180px; padding-top:2.3rem; padding-bottom:4rem; }
    header[data-testid="stHeader"] { background:rgba(243,246,244,.9); }
    h1 { letter-spacing:-.04em; font-size:clamp(2.2rem,5vw,3.7rem)!important; }
    h2, h3 { letter-spacing:-.02em; }
    [data-testid="stMetric"] {
        background:white; border:1px solid var(--line); border-radius:16px;
        padding:1.1rem 1.25rem; box-shadow:0 12px 35px rgba(27,49,40,.06);
    }
    [data-testid="stMetricLabel"] { color:var(--muted); font-weight:650; }
    [data-testid="stMetricValue"] { color:var(--ink); }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:white; border-color:var(--line)!important; border-radius:16px;
        box-shadow:0 12px 35px rgba(27,49,40,.06);
    }
    .eyebrow { color:var(--green); font-size:.77rem; font-weight:800; letter-spacing:.12em; margin:0; }
    .lead { color:var(--muted); font-size:1.05rem; margin-top:-.25rem; }
    .online { display:inline-flex; align-items:center; gap:.45rem; color:var(--muted); font-size:.86rem; }
    .online i { width:.5rem; height:.5rem; border-radius:50%; background:#48b767; display:inline-block; }
    .alert-card { border-left:4px solid #d66a22; background:#fff8f3; padding:.9rem 1rem; border-radius:0 10px 10px 0; margin:.65rem 0; }
    .alert-card.high { border-color:#c84848; background:#fff4f4; }
    .alert-card strong { display:block; margin-bottom:.2rem; }
    .alert-card small { color:var(--muted); }
    .reason { background:#e7f6eb; border-radius:10px; padding:1rem; color:#42614f; margin-top:1rem; }
    .reason b { color:var(--green); }
    .station-row { display:flex; justify-content:space-between; align-items:center; padding:.75rem 0; border-bottom:1px solid var(--line); }
    .station-row:last-child { border-bottom:0; }
    .station-row small { color:var(--muted); }
    .status { font-size:.78rem; font-weight:750; padding:.3rem .55rem; border-radius:999px; background:#e7f6eb; color:var(--green); }
    .status.review { background:#fff1e8; color:#b3571c; }
    div[role="radiogroup"] { gap:.35rem; }
    .stButton > button { background:var(--ink); color:white; border:0; border-radius:10px; font-weight:750; }
    .stButton > button:hover { background:var(--green); color:white; border:0; }
    @media (max-width: 720px) { .block-container { padding-top:1.3rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


METRICS = {
    "Temperature": {"unit": "°C", "base": 27.0, "spread": 3.0, "spike": 5.0, "color": "#267a4a"},
    "Humidity": {"unit": "%", "base": 68.0, "spread": 8.0, "spike": -14.0, "color": "#287da1"},
    "Wind speed": {"unit": " m/s", "base": 5.0, "spread": 2.5, "spike": 7.0, "color": "#8a68b7"},
}

ANOMALIES = [
    {
        "title": "Temperature spike",
        "station": "AWS-042 · Pune East",
        "detail": "+5.0°C in 10 minutes",
        "score": 98,
        "level": "high",
        "reason": "The change is much faster than normal and nearby stations do not show the same rise.",
    },
    {
        "title": "Humidity drift",
        "station": "AWS-017 · Nashik Ridge",
        "detail": "6.2% below nearby stations",
        "score": 86,
        "level": "medium",
        "reason": "The sensor has slowly moved away from the regional pattern for over three hours.",
    },
    {
        "title": "Missing wind readings",
        "station": "AWS-108 · Satara Valley",
        "detail": "11 packets missing",
        "score": 77,
        "level": "medium",
        "reason": "The station is online, but wind packets are arriving at irregular intervals.",
    },
]

STATIONS = [
    ("AWS-042", "Pune East", "Needs review"),
    ("AWS-017", "Nashik Ridge", "Healthy"),
    ("AWS-063", "Mumbai Coast", "Healthy"),
    ("AWS-108", "Satara Valley", "Healthy"),
]


@st.cache_data
def make_readings(metric_name: str) -> pd.DataFrame:
    """Create reproducible demo readings and flag outliers with a robust z-score."""
    config = METRICS[metric_name]
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("30min"), periods=48, freq="30min")
    index = np.arange(48)
    values = (
        config["base"]
        + np.sin(index / 48 * 2 * np.pi - 1.6) * config["spread"]
        + rng.normal(0, config["spread"] * 0.12, 48)
    )
    values[33] += config["spike"]

    rolling_median = pd.Series(values).rolling(9, center=True, min_periods=4).median()
    residual = pd.Series(values) - rolling_median
    median_absolute_deviation = np.median(np.abs(residual.dropna() - np.median(residual.dropna())))
    robust_z_score = 0.6745 * residual / max(median_absolute_deviation, 0.01)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values,
            "anomaly_score": robust_z_score.abs(),
            "is_anomaly": robust_z_score.abs() > 3.5,
        }
    )


def signal_chart(frame: pd.DataFrame, metric_name: str) -> go.Figure:
    config = METRICS[metric_name]
    anomalies = frame[frame["is_anomaly"]]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["value"],
            mode="lines",
            name="Sensor reading",
            line={"color": config["color"], "width": 3},
            hovertemplate=f"%{{x|%H:%M}}<br>%{{y:.1f}}{config['unit']}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["value"],
            mode="markers",
            name="AI anomaly",
            marker={"color": "#c84848", "size": 12, "line": {"color": "white", "width": 2}},
            hovertemplate="Anomaly score: %{customdata:.1f}<extra></extra>",
            customdata=anomalies["anomaly_score"],
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 8, "r": 8, "t": 16, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"showgrid": False, "title": None},
        yaxis={"gridcolor": "#e5ebe7", "title": config["unit"]},
    )
    return figure


st.markdown('<p class="eyebrow">AUTOMATIC WEATHER STATIONS</p>', unsafe_allow_html=True)
title_column, status_column = st.columns([4, 1])
with title_column:
    st.title("Weather Anomaly Detector")
    st.markdown('<p class="lead">AI checks incoming readings for sudden changes, drift, and missing data.</p>', unsafe_allow_html=True)
with status_column:
    st.markdown('<p class="online"><i></i>12 stations online</p>', unsafe_allow_html=True)

health, active, checked = st.columns(3)
health.metric("Station health", "92%", "11 healthy")
active.metric("Active anomalies", "3", "1 high priority", delta_color="inverse")
checked.metric("Readings checked", "864k", "last 24 hours")

chart_column, alert_column = st.columns([1.7, 0.8], gap="medium")
with chart_column:
    with st.container(border=True):
        st.subheader("Sensor readings")
        st.caption("AWS-042 · Pune East · last 24 hours")
        metric = st.radio("Weather metric", list(METRICS), horizontal=True, label_visibility="collapsed")
        readings = make_readings(metric)
        latest_value = readings.iloc[-1]["value"]
        st.metric(metric, f"{latest_value:.1f}{METRICS[metric]['unit']}")
        st.plotly_chart(signal_chart(readings, metric), use_container_width=True, config={"displayModeBar": False})

with alert_column:
    with st.container(border=True):
        st.subheader("Anomalies")
        st.caption("Highest priority first")
        selected_title = st.radio(
            "Choose an anomaly",
            [item["title"] for item in ANOMALIES],
            label_visibility="collapsed",
        )
        selected = next(item for item in ANOMALIES if item["title"] == selected_title)
        for item in ANOMALIES:
            css_class = "alert-card high" if item["level"] == "high" else "alert-card"
            st.markdown(
                f'<div class="{css_class}"><strong>{item["title"]} · {item["score"]}</strong>'
                f'<small>{item["station"]}<br>{item["detail"]}</small></div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="reason"><b>Why it was flagged</b><br>{selected["reason"]}</div>', unsafe_allow_html=True)

with st.container(border=True):
    header_column, button_column = st.columns([4, 1])
    with header_column:
        st.subheader("Station status")
        st.caption("Latest network check")
    with button_column:
        if st.button("Run anomaly scan", use_container_width=True):
            st.toast("Scan complete · 3 anomalies found", icon="✅")

    station_columns = st.columns(2)
    for index, (station_id, place, status) in enumerate(STATIONS):
        status_class = "status review" if status == "Needs review" else "status"
        with station_columns[index % 2]:
            st.markdown(
                f'<div class="station-row"><span><b>{station_id}</b><br><small>{place}</small></span>'
                f'<span class="{status_class}">{status}</span></div>',
                unsafe_allow_html=True,
            )

