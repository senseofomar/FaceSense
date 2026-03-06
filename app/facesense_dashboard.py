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
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# --- IMPORTS ---
from facesense.storage.db import create_session, end_active_session, get_active_session, get_connection
from facesense.core.face_detector import detect_faces


# --- UI POLISH ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* Card-like styling for metrics */
        div[data-testid="stMetric"] {
            background-color: #1E1E1E;
            border: 1px solid #333;
            padding: 10px;
            border-radius: 10px;
            color: white;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
        }
        /* Tighter padding for main container */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Custom sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #111;
        }
        </style>
    """, unsafe_allow_html=True)


# --- CONFIG ---
st.set_page_config(page_title="FaceSense AI", layout="wide", page_icon="🧠")
SNAPSHOT_PATH = os.path.join(os.getcwd(), "snapshots", "last_frame.jpg")

apply_custom_css()

# --- CHANGE 4: Gradient Hero Banner ---
st.markdown("""
    <div style='text-align:center; padding: 14px;
    background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
    border-radius: 12px; margin-bottom: 24px;
    border: 1px solid #e94560;'>
        <h1 style='color:#e94560; margin:0; font-size:2.2rem;'>🧠 FaceSense AI</h1>
        <p style='color:#aaa; margin:4px 0 0 0; font-size:1rem;'>Real-Time Emotion Recognition System</p>
    </div>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

# Global variable to store the last successful frame
_last_valid_frame = None


def load_last_snapshot(path=SNAPSHOT_PATH):
    global _last_valid_frame

    # 1. Check if file exists
    if not os.path.exists(path):
        if _last_valid_frame is not None:
            return _last_valid_frame
        return None

    # 2. Try to read (with Retries)
    for _ in range(3):
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _last_valid_frame = img
                return img
        except:
            pass
        time.sleep(0.01)

    return _last_valid_frame


def get_session_data(session_id):
    conn = get_connection()
    query = "SELECT ts, expression, confidence FROM emotion_logs WHERE session_ref_id = %s ORDER BY ts ASC"
    df = pd.read_sql(query, conn, params=(session_id,))
    conn.close()
    return df


# --- SIDEBAR ---
st.sidebar.title("🎛 Control Panel")
app_mode = st.sidebar.radio("Select Mode", ["📡 Live Monitor", "🖼️ Static Forensics", "📂 Session History"])
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

# --- PAGE 1: LIVE MONITOR (High Performance) ---
if app_mode == "📡 Live Monitor":
    st.title("📡 Live Analysis Monitor")

    # LAYOUT ADJUSTMENT: Video gets more space (Ratio 2:1)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        live_image_spot = st.empty()

    with col2:
        st.subheader("Real-time Telemetry")
        metrics_spot = st.empty()
        chart_spot = st.empty()

    last_db_update = 0
    DB_UPDATE_INTERVAL = 1.0

    while True:
        # A. FAST UPDATE
        img = load_last_snapshot()
        if img is not None:
            live_image_spot.image(img, use_container_width=True,
                                  caption=f"Latency: Real-time | {datetime.now().strftime('%H:%M:%S')}")
        else:
            live_image_spot.warning("Waiting for camera source...")

        # B. SLOW UPDATE
        if time.time() - last_db_update > DB_UPDATE_INTERVAL:
            if active_session:
                df = get_session_data(active_session[0])
                if not df.empty:
                    last = df.iloc[-1]

                    with metrics_spot.container():
                        m1, m2 = st.columns(2)
                        m1.metric("Emotion", last['expression'].upper())
                        m2.metric("Confidence", f"{float(last['confidence']) * 100:.1f}%")

                    chart = alt.Chart(df).mark_tick(thickness=3).encode(
                        x=alt.X('ts:T', title='Time', axis=alt.Axis(format='%H:%M:%S')),
                        y=alt.Y('expression:N', title='Emotion'),
                        color=alt.Color('expression:N', scale={"scheme": "category10"}),
                    ).properties(height=250, title="Emotion Timeline")
                    chart_spot.altair_chart(chart, use_container_width=True)
                else:
                    metrics_spot.info("Waiting for first detection...")
            else:
                metrics_spot.markdown("### 💤 System Idle")
                metrics_spot.write("Start a session in sidebar.")

            last_db_update = time.time()
        time.sleep(0.05)

# --- PAGE 2: STATIC FORENSICS (High Accuracy + Compact View) ---
elif app_mode == "🖼️ Static Forensics":
    st.title("🖼️ Static Image Forensics")
    st.markdown("Upload an image for deep-learning analysis.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        def resize_for_display(img, max_h=300):
            h, w = img.shape[:2]
            if h > max_h:
                scale = max_h / h
                return cv2.resize(img, (int(w * scale), max_h))
            return img

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Original")
            st.image(resize_for_display(img_rgb), use_container_width=False)

        analyze_clicked = st.button("🔍 Analyze Expression", type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner("Processing High-Res Image..."):
                from facesense.core.emotion import analyze_emotion

                faces = detect_faces(img_bgr)
                processed_img = img_rgb.copy()

                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_roi = img_bgr[y:y + h, x:x + w]
                        emotion, conf = analyze_emotion(face_roi)

                        color = (0, 255, 0)
                        cv2.rectangle(processed_img, (x, y), (x + w, y + h), color, 4)

                        label = f"{emotion.upper()}"
                        font_scale = max(0.8, processed_img.shape[1] / 1000.0)
                        thickness = max(2, int(font_scale * 2))

                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        cv2.rectangle(processed_img, (x, y - int(text_h * 1.5)), (x + text_w, y), color, -1)
                        cv2.putText(processed_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (0, 0, 0), thickness)

                    with col2:
                        st.markdown("### Processed")
                        st.image(resize_for_display(processed_img), use_container_width=False)
                        st.success(f"Detected {len(faces)} face(s).")
                else:
                    with col2:
                        st.warning("No faces detected.")

# --- PAGE 3: HISTORY ---
elif app_mode == "📂 Session History":
    st.title("📂 Session Analytics & Reports")

    try:
        conn = get_connection()
        sessions = pd.read_sql("SELECT id, session_name, start_time FROM sessions ORDER BY id DESC", conn)

        if not sessions.empty:
            selected_session_name = st.selectbox("Select a Session to Analyze:",
                                                 sessions['session_name'] + " (ID: " + sessions['id'].astype(str) + ")")
            selected_id = selected_session_name.split("ID: ")[1].replace(")", "")

            if st.button("Generate Report"):
                history_df = pd.read_sql(
                    "SELECT ts, expression, confidence FROM emotion_logs WHERE session_ref_id = %s ORDER BY ts ASC",
                    conn, params=(selected_id,)
                )

                if not history_df.empty:
                    c1, c2, c3 = st.columns(3)
                    dominant_emotion = history_df['expression'].mode()[0]
                    avg_conf = history_df['confidence'].mean()
                    if len(history_df) > 1:
                        duration = (history_df['ts'].max() - history_df['ts'].min()).seconds
                    else:
                        duration = 0

                    c1.metric("Dominant Emotion", dominant_emotion.upper())
                    c2.metric("Average Confidence", f"{avg_conf * 100:.1f}%")
                    c3.metric("Duration", f"{duration} seconds")

                    csv = history_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, f"session_{selected_id}.csv", "text/csv")

                    # --- CHANGE 2: Donut Chart + Timeline side by side ---
                    st.markdown("### 📊 Emotion Breakdown")
                    donut_col, timeline_col = st.columns([1, 2])

                    with donut_col:
                        emotion_counts = history_df['expression'].value_counts().reset_index()
                        emotion_counts.columns = ['emotion', 'count']

                        donut = alt.Chart(emotion_counts).mark_arc(innerRadius=55).encode(
                            theta=alt.Theta(field="count", type="quantitative"),
                            color=alt.Color(
                                field="emotion",
                                type="nominal",
                                scale={"scheme": "category10"},
                                legend=alt.Legend(title="Emotion")
                            ),
                            tooltip=["emotion", "count"]
                        ).properties(title="Emotion Distribution", width=220, height=220)
                        st.altair_chart(donut, use_container_width=True)

                    with timeline_col:
                        st.markdown("### 📈 Emotion Timeline")
                        history_chart = alt.Chart(history_df).mark_tick(thickness=3).encode(
                            x=alt.X('ts:T', title='Time'),
                            y=alt.Y('expression:N', title='Emotion'),
                            color=alt.Color('expression:N', scale={"scheme": "category10"})
                        ).properties(height=220).interactive()
                        st.altair_chart(history_chart, use_container_width=True)

                    with st.expander("View Raw Logs"):
                        st.dataframe(history_df)
                else:
                    st.warning("No data found.")
        else:
            st.info("No sessions found.")
        conn.close()
    except Exception as e:
        st.error(f"Database Error: {e}")