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
@st.cache_data(ttl=0.5)  # Cache for 0.5s to prevent reading file too fast
def load_last_snapshot(path=SNAPSHOT_PATH):
    if not os.path.exists(path): return None
    try:
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


# --- SIDEBAR: NAVIGATION & CONTROLS ---
st.sidebar.title("🎛 Control Panel")

# 1. NAVIGATION (Replaces Tabs)
app_mode = st.sidebar.radio("Select Mode", ["📡 Live Monitor", "🖼️ Static Forensics", "📂 Session History"])
st.sidebar.markdown("---")

# 2. SESSION CONTROLS
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

# --- PAGE 1: LIVE MONITOR (Restored Original Logic) ---
if app_mode == "📡 Live Monitor":
    st.title("📡 Live Analysis Monitor")

    col1, col2 = st.columns([1, 1.5])

    # Left Col: Camera
    with col1:
        st.subheader("Live Feed")
        img = load_last_snapshot()
        if img is not None:
            # use_container_width replaces the deprecated parameter
            st.image(img, use_container_width=True,
                     caption=f"Latency: Real-time | {datetime.now().strftime('%H:%M:%S')}")
        else:
            st.warning("Waiting for camera... (Run live.py)")

    # Right Col: Analytics
    with col2:
        st.subheader("Real-time Telemetry")
        if active_session:
            df = get_session_data(active_session[0])
            if not df.empty:
                # Metrics
                curr_emotion = df.iloc[-1]['expression']
                curr_conf = df.iloc[-1]['confidence']

                m1, m2, m3 = st.columns(3)
                m1.metric("Emotion", curr_emotion.upper())
                m2.metric("Confidence", f"{float(curr_conf) * 100:.1f}%")
                m3.metric("Data Points", len(df))

                # RESTORED: The "Tick" Chart you liked (Fast & Clean)
                chart = alt.Chart(df).mark_tick(thickness=3).encode(
                    x=alt.X('ts:T', title='Time', axis=alt.Axis(format='%H:%M:%S')),
                    y=alt.Y('expression:N', title='Emotion'),
                    color=alt.Color('expression:N', scale={"scheme": "category10"}),
                    tooltip=['ts', 'expression', 'confidence']
                ).properties(height=350, title="Emotion Timeline").interactive()

                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Session started. Waiting for first detection...")
        else:
            st.markdown("### 💤 System Idle")
            st.write("Start a session in the sidebar to begin data logging.")

    # AUTO-REFRESH (Only works in Live Mode)
    time.sleep(1)
    st.rerun()

# --- PAGE 2: STATIC FORENSICS (No Auto-Refresh) ---
elif app_mode == "🖼️ Static Forensics":
    st.title("🖼️ Static Image Forensics")
    st.markdown("Upload an image for deep-learning analysis. **(No Live Refresh)**")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original Image", use_container_width=True)

        with c2:
            if st.button("Analyze Expression"):
                with st.spinner("Loading DeepFace Model..."):
                    # Lazy Import (Keeps Live Mode Fast)
                    from facesense.core.emotion import analyze_emotion

                    faces = detect_faces(img_bgr)
                    st.success(f"Detected {len(faces)} face(s)")

                    for (x, y, w, h) in faces:
                        face_roi = img_bgr[y:y + h, x:x + w]
                        emotion, conf = analyze_emotion(face_roi)

                        # Draw visuals
                        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        label = f"{emotion.upper()} ({conf * 100:.0f}%)"
                        cv2.putText(img_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        st.write(f"**Result:** {label}")
                        st.progress(float(conf))  # Fixed float error

                    st.image(img_rgb, caption="AI Analysis Result", use_container_width=True)

# --- PAGE 3: HISTORY ---
elif app_mode == "📂 Session History":
    st.title("📂 Database Records")
    try:
        conn = get_connection()
        sessions = pd.read_sql("SELECT * FROM sessions ORDER BY id DESC LIMIT 20", conn)
        conn.close()
        st.dataframe(sessions, use_container_width=True)
    except Exception as e:
        st.error("Database unavailable.")