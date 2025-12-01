import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="facesense",
        database="facesense",
        port=3306
    )

def log_emotion(expression, confidence, bbox):
    x1, y1, x2, y2 = bbox

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO emotions (expression, confidence, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s, %s)",
        (expression, confidence, x1, y1, x2, y2)
    )

    conn.commit()
    cursor.close()
    conn.close()
