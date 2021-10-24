import sqlite3

def init_db(db_path:str, init_sql:str):
    conn = sqlite3.connect(db_path)
    with open(init_sql, 'r') as f:
        for cmd in f.read().split(';'):
            conn.execute(cmd)

    conn.close()