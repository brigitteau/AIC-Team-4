import streamlit as st
import joblib
import re

# ── Load the pipeline (vectorizer + model combined) ───────────
pipeline = joblib.load('spam_svm_model.pkl')

# ── Keep history in session state ─────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Website ───────────────────────────────────────────────────
st.title("Spam Email Detector")
st.write("Paste an email below to check if it is spam or not.")

text = st.text_area("Email text:", height=200)

if st.button("Check for Spam"):
    if text.strip():
        prediction = pipeline.predict([text])[0]
        confidence = round(pipeline.predict_proba([text]).max() * 100, 1)

        if prediction == 'Spam':
            st.error(f"🚨 SPAM — {confidence}% confidence")
        else:
            st.success(f"✅ NOT SPAM — {confidence}% confidence")

        # Save to history
        st.session_state.history.append({
            'text': text,
            'prediction': prediction,
            'confidence': confidence
        })

    else:
        st.warning("Please enter some text first.")

# ── History ───────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("Previous Checks")

    for i, entry in enumerate(reversed(st.session_state.history)):
        if entry['prediction'] == 'Spam':
            st.error(f"🚨 SPAM — {entry['confidence']}% confidence")
        else:
            st.success(f"✅ NOT SPAM — {entry['confidence']}% confidence")
        st.caption(entry['text'][:100] + '...' if len(entry['text']) > 100 else entry['text'])
        st.divider()

    