import streamlit as st

st.title("Spam Email Detector")

text = st.text_area("Paste your email text here:", height=200)

if st.button("Check for Spam"):
    if text.strip():
        st.success("Not spam!")   # placeholder for now
    else:
        st.warning("Please enter some text first.")