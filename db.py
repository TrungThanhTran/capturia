import sqlite3

class SQLIT_HANDLER(self):

with sqlite3.connect('task_queue.db') as conn:
    print("Opened database successfully")
    
with sqlite3.connect('test2.db') as conn:
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for cur in cursor:
        print(cur)