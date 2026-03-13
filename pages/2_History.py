from __future__ import annotations

import streamlit as st

from database import DBHandler


def main() -> None:
    st.set_page_config(page_title="History", layout="wide")
    st.title("Task History")

    dbhandler = DBHandler()
    table_name = "TASK_QUEUE"

    if table_name not in dbhandler.list_talbe_db():
        st.info("No history is available yet.")
        return

    records = dbhandler.check_db(table_name)
    if not records:
        st.info("No tasks found.")
        return

    st.caption(f"Showing {len(records)} task(s) from {table_name}.")
    st.dataframe(
        records,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
