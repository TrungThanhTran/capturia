from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Contact", layout="centered")
    st.title("Contact")
    st.markdown(
        """
        If you need help with transcription jobs, please contact the support team.

        - **Product:** Takenote.AI
        - **Support email:** support@takenote.ai
        - **Website:** https://www.takenote.ai/
        """
    )


if __name__ == "__main__":
    main()
