# apps/facesense_dashboard.py  (patched)
import streamlit as st
import pandas as pd
import time
import os
import cv2
from datetime import datetime
import mysql.connector

st.set_page_config(page_title="FaceSense Dashboard", layout="wide")

SNAPSHOT_PATH = os.path.join("snapshots", "last_frame.jpg")


# ---------- DB helper (reuse your db.get_connection if you want) ----------
def get_conn():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="facesense",
        database="facesense",
        port=3306
    )


# ---------- Snapshot loader with explicit debug info ----------
@st.cache_data(ttl=5)
def load_last_snapshot(path=SNAPSHOT_PATH):
    # show absolute path for easier debugging
    abs_path = os.path.abspath(path)
    if not os.path.exists(path):
        # return None but also show helpful info to the user
        st.session_state.setdefault("_snapshot_debug", f"No snapshot found at: {abs_path}")
        return None
    img = cv2.imread(path)
    if img is None:
        st.session_state.setdefault("_snapshot_debug", f"Snapshot exists but cv2.imread failed for: {abs_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.session_state.pop("_snapshot_debug", None)
    return img


# ---------- Load recent logs from DB ----------
@st.cache_data(ttl=5)
def load_logs(limit=200):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, ts, expression, confidence, x1, y1, x2, y2 FROM emotions ORDER BY ts DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame(rows)
    except Exception as e:
        # surface DB errors in UI
        st.session_state.setdefault("_db_error", str(e))
        return pd.DataFrame()


def counts_by_label(df):
    if df.empty:
        return pd.DataFrame()
    return df['expression'].value_counts().rename_axis('expression').reset_index(name='count')


# ---------- Robust force-rerun (works across versions) ----------
def force_rerun():
    # Try the direct rerun if available
    try:
        st.experimental_rerun()
        return
    except Exception:
        pass

    # If experimental_rerun isn't available, try query params (some versions)
    try:
        # read-only view of query params
        current = dict(st.query_params)
        current["_refresh_ts"] = int(time.time())
        # some streamlit versions still provide experimental_set_query_params
        try:
            st.experimental_set_query_params(**current)
            return
        except Exception:
            # fallback: modify a session_state key to trigger rerun
            st.session_state["_force_rerun_key"] = int(time.time())
            return
    except Exception:
        # final fallback: toggle a session_state key
        st.session_state["_force_rerun_key"] = int(time.time())
        return


# ---------- Layout ----------
st.title("FaceSense — Live Dashboard")
col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("Live Snapshot")
    # show debug message if snapshot missing
    img = load_last_snapshot()
    if img is not None:
        st.image(img, use_column_width=True)
    else:
        # show helpful info saved by loader
        msg = st.session_state.get("_snapshot_debug", "No snapshot found. Run FaceSense to create snapshots.")
        st.info(msg)

    if st.button("Refresh now"):
        # clear cached data and force rerun
        try:
            st.cache_data.clear()
        except Exception:
            pass
        force_rerun()

with col2:
    st.header("Recent Predictions")
    # surface DB errors if any
    db_err = st.session_state.get("_db_error")
    if db_err:
        st.error(f"DB error: {db_err}")

    df = load_logs(limit=300)
    if not df.empty:
        st.dataframe(df.head(50))
        st.markdown("### Counts")
        counts = counts_by_label(df)
        if not counts.empty:
            st.bar_chart(data=counts.set_index('expression'))
        # timeline: group by minute
        df['ts'] = pd.to_datetime(df['ts'])
        df2 = df.set_index('ts').resample('1Min').size().rename('predictions').reset_index()
        st.line_chart(df2.set_index('ts')['predictions'])
    else:
        st.info("No logs to show. Ensure DB and FaceSense are running.")

st.markdown("---")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
