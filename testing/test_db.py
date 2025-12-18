from db import get_connection

print("Starting test...")

try:
    conn = get_connection()
    print("CONNECTED SUCCESSFULLY!")
    conn.close()
except Exception as e:
    print("ERROR:", e)
