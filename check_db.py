from __future__ import annotations

from database import DBHandler


def main() -> None:
    dbhandler = DBHandler()
    for table_name in ("TASK_QUEUE", "DONE_QUEUE", "ERROR_QUEUE"):
        print(f"{table_name.lower()} = {dbhandler.check_db(table_name)}")


if __name__ == "__main__":
    main()
