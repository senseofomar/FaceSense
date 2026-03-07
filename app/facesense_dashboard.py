import sys
import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
import time
from datetime import datetime

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from facesense.storage.db import create_session, end_active_session, get_active_session, get_connection
from facesense.core.face_detector import detect_faces

# ── EMOTION COLOR PALETTE ────────────────────────────────────────────────────
# Single source of truth — matches live.py BGR colors exactly
EMOTION_COLORS = {
    "neutral":  "#00FF00",   # green
    "happy":    "#FFDC00",   # yellow
    "angry":    "#DC0000",   # red
    "sad":      "#3264DC",   # blue
    "surprise": "#00C8FF",   # cyan
    "fear":     "#B400B4",   # purple
    "disgust":  "#008C00",   # dark green
}

# Altair color scale built from the same palette
def emotion_color_scale():
    return alt.Scale(
        domain=list(EMOTION_COLORS.keys()),
        range=list(EMOTION_COLORS.values())
    )

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="FaceSense AI", layout="wide", page_icon="🧠")
SNAPSHOT_PATH = os.path.join(os.getcwd(), "snapshots", "last_frame.jpg")


def apply_custom_css():
    st.markdown("""
        <style>
        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 10px;
            padding: 14px 18px;
            color: white;
        }
        div[data-testid="stMetricLabel"]  { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
        div[data-testid="stMetricValue"]  { font-size: 1.6rem;  font-weight: 700; color: white; }

        /* Tighter main padding */
        .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; padding-left: 2rem; padding-right: 2rem; }

        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #0e0e0e; }

        /* Section dividers */
        hr { border-color: #2a2a2a; }

        /* Subheader tighter spacing */
        h3 { margin-top: 0 !important; margin-bottom: 0.4rem !important; }
        </style>
    """, unsafe_allow_html=True)


apply_custom_css()

# ── HERO BANNER ──────────────────────────────────────────────────────────────
st.markdown("""
    <div style='text-align:center; padding:16px 24px;
         background:linear-gradient(90deg,#0d0d1a,#111827,#0d1f3c);
         border-radius:10px; margin-bottom:18px; border:1px solid #e94560;'>
        <span style='font-size:2rem;font-weight:800;color:#e94560;letter-spacing:0.04em;'>
            🧠 FaceSense AI
        </span><br>
        <span style='font-size:0.9rem;color:#888;'>Real-Time Emotion Recognition System</span>
    </div>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────────────────────
_last_valid_frame = None


def load_last_snapshot(path=SNAPSHOT_PATH):
    global _last_valid_frame
    if not os.path.exists(path):
        return _last_valid_frame
    for _ in range(3):
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _last_valid_frame = img
                return img
        except Exception:
            pass
        time.sleep(0.01)
    return _last_valid_frame


def get_session_data(session_id):
    conn  = get_connection()
    query = "SELECT ts, expression, confidence FROM emotion_logs WHERE session_ref_id = %s ORDER BY ts ASC"
    df    = pd.read_sql(query, conn, params=(session_id,))
    conn.close()
    return df


def emotion_metric_color(emotion):
    return EMOTION_COLORS.get(str(emotion).lower(), "#888888")


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("🎛 Control Panel")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["📡 Live Monitor", "🖼️ Static Forensics", "📂 Session History"]
)
st.sidebar.markdown("---")

active_session = get_active_session()
if active_session:
    st.sidebar.success(f"🔴 RECORDING: {active_session[1]}")
    if st.sidebar.button("Stop Session", type="primary"):
        end_active_session()
        st.rerun()
else:
    st.sidebar.info("System Idle")
    participant_name = st.sidebar.text_input("New Session Name")
    if st.sidebar.button("Start Recording"):
        if participant_name:
            create_session(participant_name)
            st.rerun()
        else:
            st.sidebar.warning("Enter a name first.")

# ═══════════════════════════��════════════════════════════════════════════════
# PAGE 1 — LIVE MONITOR
# ════════════════════════════════════════════════════════════════════════════
if app_mode == "📡 Live Monitor":
    st.markdown("## 📡 Live Analysis Monitor")
    st.markdown("<hr style='margin:0 0 16px 0'>", unsafe_allow_html=True)

    # Layout: feed takes 65 %, telemetry panel 35 %
    feed_col, tele_col = st.columns([13, 7], gap="large")

    with feed_col:
        st.markdown("### Live Feed")
        live_image_spot = st.empty()

    with tele_col:
        st.markdown("### Telemetry")
        metrics_spot = st.empty()
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
        chart_spot   = st.empty()

    last_db_update    = 0
    DB_UPDATE_INTERVAL = 1.0

    while True:
        # A — fast: refresh image
        img = load_last_snapshot()
        if img is not None:
            live_image_spot.image(
                img, use_container_width=True,
                caption=f"Live  |  {datetime.now().strftime('%H:%M:%S')}"
            )
        else:
            live_image_spot.info("⏳ Waiting for camera feed…")

        # B — slow: refresh DB metrics + chart
        if time.time() - last_db_update > DB_UPDATE_INTERVAL:
            if active_session:
                df = get_session_data(active_session[0])
                if not df.empty:
                    last    = df.iloc[-1]
                    emotion = str(last["expression"]).lower()
                    em_hex  = emotion_metric_color(emotion)

                    with metrics_spot.container():
                        # Coloured emotion badge
                        st.markdown(
                            f"""<div style='background:#1a1a1a;border:1px solid #2a2a2a;
                                border-left:4px solid {em_hex};border-radius:10px;
                                padding:14px 18px;margin-bottom:10px;'>
                                <div style='font-size:0.75rem;color:#888;text-transform:uppercase;
                                     letter-spacing:.05em;'>Current Emotion</div>
                                <div style='font-size:1.9rem;font-weight:800;color:{em_hex};'>
                                    {emotion.upper()}
                                </div></div>""",
                            unsafe_allow_html=True
                        )
                        st.metric(
                            "Confidence",
                            f"{float(last['confidence']) * 100:.1f}%"
                        )

                    # Timeline chart with fixed emotion colours
                    chart = (
                        alt.Chart(df)
                        .mark_tick(thickness=3, size=18)
                        .encode(
                            x=alt.X("ts:T", title="Time",
                                    axis=alt.Axis(format="%H:%M:%S", labelAngle=-30)),
                            y=alt.Y("expression:N", title=None,
                                    sort=list(EMOTION_COLORS.keys())),
                            color=alt.Color(
                                "expression:N",
                                scale=emotion_color_scale(),
                                legend=None          # legend removed — colors self-evident
                            ),
                            tooltip=["expression:N",
                                     alt.Tooltip("confidence:Q", format=".0%"),
                                     alt.Tooltip("ts:T",         format="%H:%M:%S")]
                        )
                        .properties(height=220, title="Emotion Timeline")
                    )
                    chart_spot.altair_chart(chart, use_container_width=True)
                else:
                    metrics_spot.info("Waiting for first detection…")
            else:
                metrics_spot.markdown("### 💤 System Idle")
                metrics_spot.caption("Start a session in the sidebar.")

            last_db_update = time.time()

        time.sleep(0.05)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STATIC FORENSICS
# ════════════════════════════════════════════════════════════════════════════
elif app_mode == "🖼️ Static Forensics":
    st.markdown("## 🖼️ Static Image Forensics")
    st.markdown("<hr style='margin:0 0 16px 0'>", unsafe_allow_html=True)
    st.caption("Upload an image for deep-learning emotion analysis.")

    uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, 1)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        def resize_for_display(img, max_h=380):
            h, w = img.shape[:2]
            if h > max_h:
                scale = max_h / h
                return cv2.resize(img, (int(w * scale), max_h))
            return img

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### Original")
            st.image(resize_for_display(img_rgb), use_container_width=True)

        analyze_clicked = st.button(
            "🔍 Analyze Expression", type="primary", use_container_width=True
        )

        if analyze_clicked:
            with st.spinner("Running deep-learning analysis…"):
                from facesense.core.emotion import analyze_emotion

                faces         = detect_faces(img_bgr)
                processed_img = img_rgb.copy()

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_roi        = img_bgr[y:y + h, x:x + w]
                        emotion, conf   = analyze_emotion(face_roi)

                        # Use consistent emotion color
                        em_bgr  = tuple(
                            int(EMOTION_COLORS.get(emotion.lower(), "#00FF00").lstrip("#")[i:i+2], 16)
                            for i in (4, 2, 0)   # hex RGB → OpenCV BGR
                        )
                        cv2.rectangle(processed_img, (x, y), (x + w, y + h), em_bgr, 3)
                        label      = f"{emotion.upper()} {int(conf*100)}%"
                        font_scale = max(0.7, processed_img.shape[1] / 1200.0)
                        thickness  = max(2, int(font_scale * 2))
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                        )
                        cv2.rectangle(
                            processed_img, (x, y - th - 10), (x + tw + 8, y), em_bgr, -1
                        )
                        cv2.putText(
                            processed_img, label,
                            (x + 4, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness
                        )

                    with col2:
                        st.markdown("#### Processed")
                        st.image(resize_for_display(processed_img), use_container_width=True)
                        st.success(f"✅ Detected {len(faces)} face(s).")
                else:
                    with col2:
                        st.warning("No faces detected.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SESSION HISTORY
# ════════════════════════════════════════════════════════════════════════════
elif app_mode == "📂 Session History":
    st.markdown("## 📂 Session Analytics & Reports")
    st.markdown("<hr style='margin:0 0 16px 0'>", unsafe_allow_html=True)

    try:
        conn     = get_connection()
        sessions = pd.read_sql(
            "SELECT id, session_name, start_time FROM sessions ORDER BY id DESC", conn
        )

        if not sessions.empty:
            selected_name = st.selectbox(
                "Select a session to analyze:",
                sessions["session_name"] + "  (ID: " + sessions["id"].astype(str) + ")"
            )
            selected_id = selected_name.split("ID: ")[1].replace(")", "").strip()

            if st.button("Generate Report", type="primary"):
                history_df = pd.read_sql(
                    "SELECT ts, expression, confidence FROM emotion_logs "
                    "WHERE session_ref_id = %s ORDER BY ts ASC",
                    conn, params=(selected_id,)
                )

                if not history_df.empty:
                    # ── Summary metrics ──────────────────────────────────────
                    dominant = history_df["expression"].mode()[0]
                    avg_conf = history_df["confidence"].mean()
                    duration = (
                        (history_df["ts"].max() - history_df["ts"].min()).seconds
                        if len(history_df) > 1 else 0
                    )
                    dom_color = emotion_metric_color(dominant)

                    m1, m2, m3 = st.columns(3, gap="medium")
                    m1.markdown(
                        f"""<div style='background:#1a1a1a;border:1px solid #2a2a2a;
                            border-left:4px solid {dom_color};border-radius:10px;padding:14px 18px;'>
                            <div style='font-size:.75rem;color:#888;text-transform:uppercase;
                                 letter-spacing:.05em;'>Dominant Emotion</div>
                            <div style='font-size:1.6rem;font-weight:800;color:{dom_color};'>
                                {dominant.upper()}</div></div>""",
                        unsafe_allow_html=True
                    )
                    m2.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")
                    m3.metric("Duration",       f"{duration}s")

                    st.markdown("<div style='margin:16px 0 4px 0'></div>", unsafe_allow_html=True)
                    csv = history_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download CSV", csv,
                        f"session_{selected_id}.csv", "text/csv"
                    )

                    # ── Charts ───────────────────────────────────────────────
                    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
                    donut_col, timeline_col = st.columns([5, 7], gap="large")

                    with donut_col:
                        st.markdown("#### Emotion Distribution")
                        counts = (
                            history_df["expression"]
                            .value_counts()
                            .reset_index()
                            .rename(columns={"index": "emotion", "expression": "count"})
                        )
                        # Ensure column names are right regardless of pandas version
                        if "emotion" not in counts.columns:
                            counts.columns = ["emotion", "count"]

                        donut = (
                            alt.Chart(counts)
                            .mark_arc(innerRadius=60, outerRadius=110)
                            .encode(
                                theta=alt.Theta("count:Q"),
                                color=alt.Color(
                                    "emotion:N",
                                    scale=emotion_color_scale(),
                                    legend=alt.Legend(
                                        title=None,
                                        orient="bottom",
                                        columns=2,
                                        labelFontSize=12,
                                        symbolSize=120
                                    )
                                ),
                                tooltip=["emotion:N", "count:Q"]
                            )
                            .properties(width=260, height=260)
                        )
                        st.altair_chart(donut, use_container_width=True)

                    with timeline_col:
                        st.markdown("#### Emotion Timeline")
                        timeline = (
                            alt.Chart(history_df)
                            .mark_tick(thickness=3, size=20)
                            .encode(
                                x=alt.X("ts:T", title="Time",
                                        axis=alt.Axis(format="%H:%M:%S", labelAngle=-30)),
                                y=alt.Y("expression:N", title=None,
                                        sort=list(EMOTION_COLORS.keys())),
                                color=alt.Color(
                                    "expression:N",
                                    scale=emotion_color_scale(),
                                    legend=alt.Legend(
                                        title=None,
                                        orient="bottom",
                                        columns=4,
                                        labelFontSize=12,
                                        symbolSize=120
                                    )
                                ),
                                tooltip=["expression:N",
                                         alt.Tooltip("confidence:Q", format=".0%"),
                                         alt.Tooltip("ts:T",         format="%H:%M:%S")]
                            )
                            .properties(height=260)
                            .interactive()
                        )
                        st.altair_chart(timeline, use_container_width=True)

                    with st.expander("📋 View Raw Logs"):
                        st.dataframe(history_df, use_container_width=True)
                else:
                    st.warning("No data found for this session.")
        else:
            st.info("No sessions found.")

        conn.close()

    except Exception as e:
        st.error(f"Database Error: {e}")