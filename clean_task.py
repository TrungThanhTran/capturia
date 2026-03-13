from __future__ import annotations

from database import DBHandler


def main() -> None:
    dbhandler = DBHandler()
    print(dbhandler.delete_all_db(tb_name="DONE_QUEUE"))


if __name__ == "__main__":
    main()
