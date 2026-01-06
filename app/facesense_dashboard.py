import streamlit as st
import pandas as pd
import time
import os
import cv2
import mysql.connector
import altair as alt  # For beautiful charts
from datetime import datetime

# Import your DB functions
from facesense.storage.db import create_session, end_active_session, get_active_session

# --- PAGE CONFIG ---
st.set_page_config(page_title="FaceSense AI", layout="wide")

# --- SNAPSHOT CONFIGURATION ---
# This path must match where live.py saves images
SNAPSHOT_PATH = os.path.join(os.getcwd(), "snapshots", "last_frame.jpg")


@st.cache_data(ttl=1)  # Cache for 1 second to force frequent reloads
def load_last_snapshot(path=SNAPSHOT_PATH):
    """
    Loads the most recent frame saved by live.py.
    Handles file locking and missing file errors gracefully.
    """
    abs_path = os.path.abspath(path)

    # Check if file exists
    if not os.path.exists(path):
        return None

    # Read image
    img = cv2.imread(path)
    if img is None:
        return None

    # Convert BGR (OpenCV) to RGB (Streamlit)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# --- DATABASE HELPER ---
def get_data_for_session(session_id):
    """
    Fetches emotion logs specifically for the active session.
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="facesense",
            database="facesense",
            port=3306
        )
        query = """
                SELECT ts, expression, confidence
                FROM emotion_logs
                WHERE session_ref_id = %s
                ORDER BY ts ASC \
                """
        df = pd.read_sql(query, conn, params=(session_id,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()


# --- SIDEBAR: CONTROLS ---
st.sidebar.title("🎛 Control Panel")
active_session = get_active_session()

if active_session:
    # If a session is running, show STOP button
    st.sidebar.success(f"🔴 RECORDING: {active_session[1]}")
    if st.sidebar.button("Stop Session", type="primary"):
        end_active_session()
        st.rerun()
else:
    # If idle, show START controls
    st.sidebar.info("System is Idle")
    participant_name = st.sidebar.text_input("Subject Name / Session ID")
    if st.sidebar.button("Start Recording"):
        if participant_name:
            create_session(participant_name)
            st.rerun()
        else:
            st.sidebar.warning("Please enter a name first.")

# --- MAIN LAYOUT ---
st.title("🧠 FaceSense: Emotion Analytics")

col1, col2 = st.columns([1, 1.5])

# === COLUMN 1: LIVE VIDEO FEED ===
with col1:
    st.subheader("Live Feed")

    # Load the latest image using the function we integrated
    img = load_last_snapshot()

    if img is not None:
        st.image(img, use_column_width=True)
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    else:
        st.warning("Waiting for camera... (Ensure live.py is running)")
        st.info(f"Looking for snapshot at: {SNAPSHOT_PATH}")

    # Manual refresh button (just in case)
    if st.button("Refresh Feed"):
        st.cache_data.clear()
        st.rerun()

# === COLUMN 2: ANALYTICS ===
with col2:
    st.subheader("Real-time Analysis")

    if active_session:
        # Fetch live data for THIS session
        df = get_data_for_session(active_session[0])

        if not df.empty:
            # 1. Metric Cards (Top Row)
            curr_emotion = df.iloc[-1]['expression']
            curr_conf = df.iloc[-1]['confidence']
            total_points = len(df)

            m1, m2, m3 = st.columns(3)
            m1.metric("Current Emotion", curr_emotion.upper())
            m2.metric("Confidence", f"{curr_conf * 100:.1f}%")
            m3.metric("Data Points", total_points)

            # 2. CHART: Stacked Area Chart (The "Wow" factor)
            # This creates a timeline of emotions detected
            chart = alt.Chart(df).mark_tick(thickness=3).encode(
                x=alt.X('ts:T', title='Time', axis=alt.Axis(format='%H:%M:%S')),
                y=alt.Y('expression:N', title='Emotion'),
                color=alt.Color('expression:N', scale={"scheme": "category10"}, legend=alt.Legend(title="Emotion")),
                tooltip=['ts', 'expression', 'confidence']
            ).properties(
                height=350,
                title="Emotion Timeline"
            ).interactive()

            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("Session started. Waiting for first detection...")
            time.sleep(1)
            st.rerun()
    else:
        st.markdown("### 💤 System Idle")
        st.write("Start a session in the sidebar to begin analyzing emotions.")
        st.write("Current data will be saved to MySQL for later review.")

# --- AUTO-REFRESH LOOP ---
# This keeps the dashboard updating automatically every second
time.sleep(1)
st.rerun()