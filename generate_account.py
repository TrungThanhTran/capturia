from __future__ import annotations

import streamlit_authenticator as stauth


def generate_hashed_passwords(passwords: list[str]) -> list[str]:
    return stauth.Hasher(passwords).generate()


if __name__ == "__main__":
    print(generate_hashed_passwords(["Takenote"]))
