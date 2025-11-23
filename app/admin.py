# Admin dashboard logic
import streamlit as st
import requests
import pandas as pd

def render_admin(admin_url: str):
    st.header("Admin Dashboard")
    st.info("Shows uploaded documents and collected leads (demo).")
    try:
        r = requests.get(admin_url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.error(f"Failed to fetch admin data: {e}")
        return

    leads = data.get("leads", [])
    uploads = data.get("uploads", [])

    st.subheader("Leads / Contacts")
    if leads:
        df = pd.DataFrame(leads)
        st.dataframe(df)
    else:
        st.write("No leads collected yet.")

    st.subheader("Indexed uploads")
    if uploads:
        df2 = pd.json_normalize(uploads)
        st.dataframe(df2)
    else:
        st.write("No uploads indexed yet.")