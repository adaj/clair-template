import os
from pathlib import Path
import streamlit as st
import requests
from dotenv import load_dotenv

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
# Ensure the CLAIR_URL environment variable is set.
CLAIR_URL = os.environ.get('CLAIR_URL')
assert CLAIR_URL and isinstance(CLAIR_URL, str), "Please set the env variable CLAIR_URL with a running clair http app."
CLAIR_URL = CLAIR_URL.rstrip('/')
print("CLAIR_URL: ", CLAIR_URL)

BASE_URL = f"{CLAIR_URL}/blacklist"

def get_blacklist():
    """Fetches the blacklisted users from the API"""
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.warning(f"Error fetching blacklist: {e}")
        return []

def add_to_blacklist(username):
    """Adds a user to the blacklist via the API"""
    try:
        response = requests.post(f"{BASE_URL}/{username}")
        response.raise_for_status()
        st.success(f"User {username} added to blacklist!")
    except requests.RequestException as e:
        st.warning(f"Error adding user to blacklist: {e}")

st.title("Blacklist Management")

# Add user to blacklist
st.subheader("Add User to Blacklist")
username = st.text_input("Username")
if st.button("Submit"):
    if username:
        add_to_blacklist(username)
    else:
        st.warning("Please provide a username!")

# Fetch blacklisted users
st.subheader("Blacklisted users:")
if st.button("Refresh"):
    blacklist = get_blacklist()
    for user in blacklist:
        st.write(user)

if __name__ == "__main__":
    pass  # this is here just to structure the code as a typical Streamlit app
