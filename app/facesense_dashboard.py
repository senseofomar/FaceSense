# apps/facesense_dashboard.py
import streamlit as st
import pandas as pd
import time
import os
import cv2
from datetime import datetime
import mysql.connector

st.set_page_config(page_title="FaceSense Dashboard", layout="wide")

# DB connection helper (reuse your db.get_connection if preferred)
def get_conn():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="facesense",
        database="facesense",
        port=3306
    )

@st.cache_data(ttl=5)
def load_last_snapshot(path="snapshots/last_frame.jpg"):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

@st.cache_data(ttl=5)
def load_logs(limit=200):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, ts, expression, confidence, x1, y1, x2, y2 FROM emotion_logs ORDER BY ts DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()

def counts_by_label(df):
    if df.empty:
        return pd.DataFrame()
    return df['expression'].value_counts().rename_axis('expression').reset_index(name='count')

st.title("FaceSense — Live Dashboard")
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("Live Snapshot")
    img = load_last_snapshot()
    if img is not None:
        st.image(img, use_column_width=True)
    else:
        st.info("No snapshot found. Run FaceSense to create snapshots.")

    if st.button("Refresh now"):
        st.cache_data.clear()
        st.experimental_rerun()

with col2:
    st.header("Recent Predictions")
    df = load_logs(limit=300)
    if not df.empty:
        st.dataframe(df.head(50))
        st.markdown("### Counts")
        counts = counts_by_label(df)
        st.bar_chart(data=counts.set_index('expression'))
        # timeline: group by minute
        df['ts'] = pd.to_datetime(df['ts'])
        df2 = df.set_index('ts').resample('1Min').size().rename('predictions').reset_index()
        st.line_chart(df2.set_index('ts')['predictions'])
    else:
        st.info("No logs to show. Ensure DB and FaceSense are running.")

st.markdown("---")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
