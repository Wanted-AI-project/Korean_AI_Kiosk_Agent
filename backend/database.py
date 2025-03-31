import sqlite3

def get_db_connection():
    conn = sqlite3.connect("data/Comfile_Coffee_DB.db")
    conn.row_factory = sqlite3.Row  # dict 형태로 반환
    return conn
