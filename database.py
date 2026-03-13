from __future__ import annotations

import re
import sqlite3
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DBHandler:
    def __init__(self, db_name: str = "DB_TASK.db") -> None:
        self.db_name = db_name

    def _validate_table_name(self, tb_name: str) -> str:
        if not _IDENTIFIER_RE.match(tb_name):
            raise ValueError(f"Invalid table name: {tb_name}")
        return tb_name

    def connect_db(self) -> None:
        with sqlite3.connect(self.db_name):
            pass

    def list_talbe_db(self) -> list[str]:
        query = 'SELECT name FROM sqlite_master WHERE type="table"'
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)
            return [row[0] for row in cursor.fetchall()]

    def create_table(self, tb_name: str) -> None:
        table_name = self._validate_table_name(tb_name)
        query = f"""
            CREATE TABLE {table_name} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                TASKID TEXT NOT NULL,
                FILE_PATH TEXT NOT NULL,
                USER TEXT NOT NULL,
                EMAIL TEXT NOT NULL,
                TIME TEXT,
                STATUS INT
            )
        """
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(query)
            conn.commit()

    def writeinfo_db(self, tb_name: str, task_id: str, file_name: str, user: str, email: str, time: str, status: int = 1) -> None:
        table_name = self._validate_table_name(tb_name)
        query = f"INSERT INTO {table_name} (TASKID, FILE_PATH, USER, EMAIL, TIME, STATUS) VALUES (?, ?, ?, ?, ?, ?)"
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(query, (task_id, file_name, user, email, time, status))
            conn.commit()

    def query_db(self, query: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()

    def query_db_min(self, tb_name: str) -> tuple[Any, ...]:
        table_name = self._validate_table_name(tb_name)
        query = f"SELECT * FROM {table_name} WHERE ID = (SELECT MIN(ID) FROM {table_name})"
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)
            row = cursor.fetchone()
        if row is None:
            raise LookupError(f"Table {table_name} is empty")
        return row

    def delete_task_db(self, done_id: int, tb_name: str = "TASK_QUEUE") -> str:
        table_name = self._validate_table_name(tb_name)
        try:
            query = f"DELETE FROM {table_name} WHERE ID = ?"
            with sqlite3.connect(self.db_name) as conn:
                conn.execute(query, (done_id,))
                conn.commit()
            return "successfully delete done_task!"
        except Exception as exc:
            return str(exc)

    def delete_all_db(self, tb_name: str = "TASK_QUEUE") -> str:
        table_name = self._validate_table_name(tb_name)
        try:
            query = f"DELETE FROM {table_name}"
            with sqlite3.connect(self.db_name) as conn:
                conn.execute(query)
                conn.commit()
            return "successfully delete done_task!"
        except Exception as exc:
            return str(exc)

    def get_len_table_db(self, tb_name: str = "TASK_QUEUE") -> int:
        table_name = self._validate_table_name(tb_name)
        query = f"SELECT COUNT(*) FROM {table_name}"
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)
            return int(cursor.fetchone()[0])

    def check_db(self, tb_name: str = "TASK_QUEUE") -> list[tuple[Any, ...]]:
        table_name = self._validate_table_name(tb_name)
        query = f"SELECT * FROM {table_name}"
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(query)
            return cursor.fetchall()

    def get_db_by_user(self, tb_name: str, user_name: str) -> list[tuple[Any, ...]]:
        table_name = self._validate_table_name(tb_name)
        query = f"SELECT * FROM {table_name} WHERE USER = ?"
        return self.query_db(query, (user_name,))
