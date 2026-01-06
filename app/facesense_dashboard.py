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

# --- CONFIG ---
st.set_page_config(page_title="FaceSense AI", layout="wide", page_icon="🧠")
SNAPSHOT_PATH = os.path.join(os.getcwd(), "snapshots", "last_frame.jpg")


# --- HELPER FUNCTIONS ---

# REMOVED @st.cache_data for this specific function.
# Caching prevents real-time 30FPS playback because it holds old frames.
def load_last_snapshot(path=SNAPSHOT_PATH):
    if not os.path.exists(path): return None
    try:
        # Use simple file read - OS handles buffering efficiently
        img = cv2.imread(path)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return None


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

    col1, col2 = st.columns([1, 1.5])

    # 1. Create PLACEHOLDERS once.
    with col1:
        st.subheader("Live Feed")
        live_image_spot = st.empty()

    with col2:
        st.subheader("Real-time Telemetry")
        metrics_spot = st.empty()
        chart_spot = st.empty()

    # 2. THE SMOOTH LOOP
    # This loop keeps running to update the image fast,
    # but only fetches DB data occasionally.

    last_db_update = 0
    DB_UPDATE_INTERVAL = 1.0  # Update graph every 1 second

    while True:
        # A. FAST UPDATE: Camera Feed (Every 0.05s)
        img = load_last_snapshot()
        if img is not None:
            live_image_spot.image(img, use_container_width=True,
                                  caption=f"Latency: Real-time | {datetime.now().strftime('%H:%M:%S')}")
        else:
            live_image_spot.warning("Waiting for camera source...")

        # B. SLOW UPDATE: Database & Graph (Every 1.0s)
        # We throttle this to prevent UI freezing and DB overload
        if time.time() - last_db_update > DB_UPDATE_INTERVAL:
            if active_session:
                df = get_session_data(active_session[0])
                if not df.empty:
                    last = df.iloc[-1]

                    # Update Metrics
                    with metrics_spot.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Emotion", last['expression'].upper())
                        m2.metric("Confidence", f"{float(last['confidence']) * 100:.1f}%")
                        m3.metric("Data Points", len(df))

                    # Update Chart
                    chart = alt.Chart(df).mark_tick(thickness=3).encode(
                        x=alt.X('ts:T', title='Time', axis=alt.Axis(format='%H:%M:%S')),
                        y=alt.Y('expression:N', title='Emotion'),
                        color=alt.Color('expression:N', scale={"scheme": "category10"}),
                    ).properties(height=350, title="Emotion Timeline")
                    chart_spot.altair_chart(chart, use_container_width=True)
                else:
                    metrics_spot.info("Waiting for first detection...")
            else:
                metrics_spot.markdown("### 💤 System Idle")
                metrics_spot.write("Start a session in the sidebar to begin data logging.")

            last_db_update = time.time()  # Reset timer

        # C. CONTROL FRAME RATE
        # Sleep slightly to release CPU for sidebar interaction
        time.sleep(0.05)

    # --- PAGE 2 Static ---
elif app_mode == "🖼️ Static Forensics":
    st.title("🖼️ Static Image Forensics")
    st.markdown("Upload an image for deep-learning analysis.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with c2:
            if st.button("Analyze Expression"):
                with st.spinner("Processing..."):
                    from facesense.core.emotion import analyze_emotion

                    faces = detect_faces(img_bgr)
                    st.success(f"Detected {len(faces)} face(s)")
                    for (x, y, w, h) in faces:
                        face_roi = img_bgr[y:y + h, x:x + w]
                        emotion, conf = analyze_emotion(face_roi)
                        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        label = f"{emotion.upper()} ({conf * 100:.0f}%)"
                        cv2.putText(img_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        st.write(f"**Result:** {label}")
                        st.progress(float(conf))
                    st.image(img_rgb, caption="Processed", use_container_width=True)

# --- PAGE 3: HISTORY ---
elif app_mode == "📂 Session History":
    st.title("📂 Session Analytics & Reports")

    try:
        conn = get_connection()
        # Fetch session list for the dropdown
        sessions = pd.read_sql("SELECT id, session_name, start_time FROM sessions ORDER BY id DESC", conn)

        if not sessions.empty:
            # Create a dropdown to select a session
            selected_session_name = st.selectbox("Select a Session to Analyze:",
                                                 sessions['session_name'] + " (ID: " + sessions['id'].astype(str) + ")")

            # Extract the ID from the string
            selected_id = selected_session_name.split("ID: ")[1].replace(")", "")

            if st.button("Generate Report"):
                # Fetch data for this specific past session
                history_df = pd.read_sql(
                    "SELECT ts, expression, confidence FROM emotion_logs WHERE session_ref_id = %s ORDER BY ts ASC",
                    conn,
                    params=(selected_id,)
                )

                if not history_df.empty:
                    # Metrics
                    c1, c2, c3 = st.columns(3)
                    dominant_emotion = history_df['expression'].mode()[0]
                    avg_conf = history_df['confidence'].mean()

                    # Fix for single-row duration bug
                    if len(history_df) > 1:
                        duration = (history_df['ts'].max() - history_df['ts'].min()).seconds
                    else:
                        duration = 0

                    c1.metric("Dominant Emotion", dominant_emotion.upper())
                    c2.metric("Average Confidence", f"{avg_conf * 100:.1f}%")
                    c3.metric("Duration", f"{duration} seconds")

                    # CSV Export
                    csv = history_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Session Data (CSV)",
                        data=csv,
                        file_name=f"session_{selected_id}_report.csv",
                        mime="text/csv",
                    )

                    # Chart
                    st.markdown("### 📈 Emotion Timeline")
                    history_chart = alt.Chart(history_df).mark_tick(thickness=3).encode(
                        x=alt.X('ts:T', title='Time'),
                        y=alt.Y('expression:N', title='Emotion'),
                        color=alt.Color('expression:N', scale={"scheme": "category10"})
                    ).properties(height=350).interactive()

                    st.altair_chart(history_chart, use_container_width=True)

                    with st.expander("View Raw Data Logs"):
                        st.dataframe(history_df)
                else:
                    st.warning("No data found for this session.")
            else:
                st.info("No sessions found in database.")

            conn.close()

    except Exception as e:
        st.error(f"Database Error: {e}")