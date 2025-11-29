# db.py
import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="face",
        database="facesense"
    )

def log_emotion(emotion, confidence, bbox):
    x1, y1, x2, y2 = bbox

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO emotion_logs (emotion, confidence, x1, y1, x2, y2)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (emotion, confidence, x1, y1, x2, y2)

    cursor.execute(sql, values)
    conn.commit()

    cursor.close()
    conn.close()
