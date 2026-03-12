import os
import MySQLdb
import datetime
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)


def get_connection():
    return MySQLdb.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        passwd=os.getenv("DB_PASS", "facesense"),
        db=os.getenv("DB_NAME", "facesense"),
        port=int(os.getenv("DB_PORT", "3306"))
    )


# ── Session Management ────────────────────────────────────────────────────────

def create_session(name):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET is_active=0 WHERE is_active=1")
    cursor.execute("INSERT INTO sessions (session_name) VALUES (%s)", (name,))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def set_session_video_path(session_id, video_path):
    """Called by live.py once the video file is created, stores the path in DB."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        # Add video_path column if it doesn't exist yet (safe to run repeatedly)
        cursor.execute("""
            ALTER TABLE sessions
            ADD COLUMN IF NOT EXISTS video_path VARCHAR(512) DEFAULT NULL
        """)
        cursor.execute(
            "UPDATE sessions SET video_path = %s WHERE id = %s",
            (video_path, session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB WARNING (video path): {e}")


def end_active_session():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET is_active=0, end_time=NOW() WHERE is_active=1")
    conn.commit()
    conn.close()


def get_active_session():
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, session_name FROM sessions WHERE is_active=1 ORDER BY id DESC LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()
    return row   # (id, name)  or  None


def set_session_video_path(session_id, video_path):
    """Called by live.py once the video file is created, stores the path in DB."""
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        # Column already exists in DB — no need to ALTER TABLE here
        cursor.execute(
            "UPDATE sessions SET video_path = %s WHERE id = %s",
            (video_path, session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB WARNING (video path): {e}")

# ── Emotion Logging ───────────────────────────────────────────────────────────

def log_emotion(expression, confidence, bbox, session_ref_id=None):
    try:
        conn   = get_connection()
        cursor = conn.cursor()
        x1, y1, x2, y2 = map(int, bbox)
        confidence      = float(confidence)

        if session_ref_id is None:
            active         = get_active_session()
            session_ref_id = active[0] if active else None

        cursor.execute(
            """INSERT INTO emotion_logs
               (expression, confidence, x1, y1, x2, y2, session_ref_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (expression, confidence, x1, y1, x2, y2, session_ref_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB ERROR:", e)