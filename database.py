import sqlite3


class DBHandler():
    def __init__(self, db_name='DB_TASK.db') -> None:
        self.db_name = db_name

    def connect_db(self):
        with sqlite3.connect(self.db_name) as conn:
            print("Opened database successfully")

    def list_talbe_db(self):
        with sqlite3.connect(self.db_name) as conn:
            query = f'''    
                    SELECT name FROM sqlite_master WHERE type="table"
                    '''
            cursor = conn.execute(query)
        return [curr[0] for curr in cursor]

    def create_table(self, tb_name):
        # TASK_QUEUE
        # Create table for the database
        query = f'''
                    CREATE TABLE {tb_name}
                    (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    TASKID           TEXT      NOT NULL,
                    FILE_PATH        TEXT     NOT NULL,
                    USER             TEXT    NOT NULL,
                    EMAIL            TEXT    NOT NULL,
                    TIME             TEXT,
                    STATUS           INT);
                    '''
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(query)
        print("Table created successfully")

    def writeinfo_db(self, tb_name, task_id, file_name, user, email, time, status=1):
        """
        status = 0: not done, 1: success done, 2: re-try
        """
        query = f'''
                INSERT INTO {tb_name} (TASKID,FILE_PATH,USER,EMAIL,TIME,STATUS) \
                VALUES ('{task_id}','{file_name}','{user}','{email}','{time}',{status})'''
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(query)

    def query_db(self, query):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)
        return cursor.fetchall()

    def query_db_min(self, tb_name):
        query = f'''
                    SELECT * \
                    FROM {tb_name} \
                    WHERE  ID = (SELECT MIN(ID) \
                    FROM {tb_name})
                '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)
        return cursor.fetchall()[0]
    
    def delete_task_db(self, done_id, tb_name='TASK_QUEUE'):
        try:
            query = f'''
                        DELETE FROM {tb_name} WHERE ID = {done_id}
                    '''
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.execute(query)
            return "successfully delete done_task!"
        except Exception as e:
            return e
        
    def delete_all_db(self, tb_name='TASK_QUEUE'):
        try:
            query = f'''
                        DELETE FROM {tb_name}
                    '''
            with sqlite3.connect(self.db_name) as conn:
                cursor = conn.execute(query)
            return "successfully delete done_task!"
        except Exception as e:
            return e
    
    def get_len_table_db(self, tb_name='TASK_QUEUE'):
        query = f'''
                    SELECT * FROM {tb_name}
                '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)  
        return len(cursor.fetchall()) 
    
    def check_db(self, tb_name='TASK_QUEUE'):
        query = f'''
                    SELECT * FROM {tb_name}
                '''
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)  
        return cursor.fetchall()
    
    def get_db_by_user(self, tb_name, user_name):
        query = f'''
                    SELECT * FROM {tb_name} \
                        WHERE USER = '{user_name}'
                '''
        return self.query_db(query)
        
