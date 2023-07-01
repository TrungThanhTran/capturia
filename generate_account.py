import streamlit_authenticator as stauth
hashed_passwords = stauth.Hasher(['Takenote']).generate()
print(hashed_passwords)