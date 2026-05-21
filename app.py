import streamlit as st
import joblib
import re

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&family=Nunito:wght@400;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], .stTextArea, .stButton,
.stAlert, .stCaption, label, input, textarea, button,
h1, h2, h3, h4, h5, h6, p, div, span {
    font-family: 'Nunito', sans-serif !important;
}

h1, h2, h3, [data-testid="stHeadingWithActionElements"] * {
    font-family: 'Fredoka', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        160deg,
        #FFFFFF 0%,
        #FFF5E1 25%,
        #FFDAB9 55%,
        #FFB6A3 80%,
        #FF8C75 100%
    );
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: rgba(255, 220, 185, 0.55);
}
</style>
""", unsafe_allow_html=True)

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

    