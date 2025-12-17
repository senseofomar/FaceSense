import MySQLdb


def get_connection():
    return MySQLdb.connect(
    host="localhost",
    user="root",
    passwd="facesense",
    db="facesense",
    port=3306
)


def log_emotion(expression, confidence, bbox, session_id = None):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        x1, y1, x2, y2 = bbox

        # Cast to native Python types
        # - Fixed the DB error: Python type numpy.int64 cannot be converted
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        confidence = float(confidence)

        query = """
        INSERT INTO emotion_logs (expression, confidence, 
                                  x1, y1, x2, y2, session_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(query, (expression, confidence,
                               x1, y1, x2, y2, session_id))
        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print("DB ERROR:", e)
