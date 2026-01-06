import MySQLdb
import datetime


def get_connection():
    return MySQLdb.connect(host="localhost", user="root", passwd="facesense", db="facesense", port=3306)


# --- NEW: Session Management ---
def create_session(name):
    conn = get_connection()
    cursor = conn.cursor()
    # Deactivate any previous active sessions first (fail-safe)
    cursor.execute("UPDATE sessions SET is_active=0 WHERE is_active=1")

    query = "INSERT INTO sessions (session_name) VALUES (%s)"
    cursor.execute(query, (name,))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def end_active_session():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET is_active=0, end_time=NOW() WHERE is_active=1")
    conn.commit()
    conn.close()


def get_active_session():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, session_name FROM sessions WHERE is_active=1 ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row  # Returns (id, name) or None


# --- UPDATED: Logging ---
def log_emotion(expression, confidence, bbox, session_ref_id=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        x1, y1, x2, y2 = map(int, bbox)
        confidence = float(confidence)

        # If no session provided, try to find active one
        if session_ref_id is None:
            active = get_active_session()
            session_ref_id = active[0] if active else None

        query = """
                INSERT INTO emotion_logs (expression, confidence, x1, y1, x2, y2, session_ref_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s) \
                """
        cursor.execute(query, (expression, confidence, x1, y1, x2, y2, session_ref_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB ERROR:", e)