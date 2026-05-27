import streamlit as st
import joblib
import re
import base64

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&family=Nunito:wght@400;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], .stTextArea, .stButton,
.stAlert, .stCaption, label, input, textarea, button,
h1, h2, h3, h4, h5, h6, p, div, span {
    font-family: 'Nunito', sans-serif !important;
    color: black !important;
}

h1, h2, h3, [data-testid="stHeadingWithActionElements"] * {
    font-family: 'Fredoka', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    color: black !important;
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


/* textarea: white background, thick black border, white text */
textarea {
    color: black !important;
    background-color: white !important;
    border: 3px solid black !important;
    border-radius: 8px !important;
}

/* "Email text:" label bigger */
[data-testid="stTextArea"] label p {
    font-size: 1.15rem !important;
}

/* instructions text slightly bigger */
[data-testid="stMarkdownContainer"] p {
    font-size: 1.05rem !important;
}

/* three-star loading row next to label */
.star-ring {
    display: inline-flex;
    gap: 4px;
    vertical-align: middle;
}
.star-ring span {
    font-size: 12px;
    color: black !important;
    animation: star-pulse 1.2s ease-in-out infinite;
}
.star-ring span:nth-child(2) { animation-delay: 0.2s; }
.star-ring span:nth-child(3) { animation-delay: 0.4s; }
@keyframes star-pulse {
    0%, 100% { transform: translateY(0);  opacity: 0.3; }
    50%       { transform: translateY(-4px); opacity: 1; }
}
.email-label {
    font-family: 'Nunito', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: black;
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}

/* "Check for Spam" button text white + hover float */
[data-testid="stButton"] button p {
    color: white !important;
}
[data-testid="stButton"] button {
    color: white !important;
    border: 2px solid black !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.18) !important;
    color: black !important;
}
[data-testid="stButton"] button:hover p {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load the pipeline (vectorizer + model combined) ───────────
pipeline = joblib.load('spam_svm_model.pkl')

# ── Pre-encode both logos ──────────────────────────────────────
with open("/Users/aditid/Downloads/musubi_no_bg.png", "rb") as f:
    happy_b64 = base64.b64encode(f.read()).decode()
with open("/Users/aditid/Downloads/sad_musubi.png", "rb") as f:
    sad_b64 = base64.b64encode(f.read()).decode()

# ── Keep history in session state ─────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Website ───────────────────────────────────────────────────
col_title, col_logo = st.columns([5, 1])
with col_title:
    st.title("Spam Email Detector")
    st.write("Paste an email below to check if it is spam or not.")
with col_logo:
    st.markdown(f"""
<img src="data:image/png;base64,{happy_b64}"
     style="width:100px;margin-top:8px;">
""", unsafe_allow_html=True)

st.markdown("""
<div class="email-label">
  Email text:
  <span class="star-ring">
    <span>★</span><span>★</span><span>★</span>
  </span>
</div>
""", unsafe_allow_html=True)
text = st.text_area("Email text:", height=200, label_visibility="collapsed")

if st.button("Check for Spam"):
    if text.strip():
        prediction = pipeline.predict([text])[0]
        confidence = round(pipeline.predict_proba([text]).max() * 100, 1)

        if prediction == 'Spam':
            st.markdown(f"""
<div style="background:#ffd2d2;border:1px solid #f44;border-radius:8px;padding:12px 16px;position:relative;min-height:64px;display:flex;align-items:center;">
  <span style="font-weight:900;font-size:1.1rem;color:#900;">! SPAM — {confidence}% confidence</span>
  <img src="data:image/png;base64,{sad_b64}" style="position:absolute;right:-18px;top:-28px;width:90px;">
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:#d4edda;border:1px solid #28a745;border-radius:8px;padding:12px 16px;position:relative;min-height:64px;display:flex;align-items:center;">
  <span style="font-weight:900;font-size:1.1rem;color:#155724;">✔ NOT SPAM — {confidence}% confidence</span>
  <img src="data:image/png;base64,{happy_b64}" style="position:absolute;right:-18px;top:-28px;width:90px;">
</div>
""", unsafe_allow_html=True)

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
            st.markdown(f"""
<div style="background:#ffd2d2;border:1px solid #f44;border-radius:8px;padding:12px 16px;position:relative;min-height:64px;display:flex;align-items:center;">
  <span style="font-weight:900;font-size:1.1rem;color:#900;">! SPAM — {entry['confidence']}% confidence</span>
  <img src="data:image/png;base64,{sad_b64}" style="position:absolute;right:-18px;top:-28px;width:90px;">
</div>
""", unsafe_allow_html=True)
        else:
            st.success(f"✔ NOT SPAM — {entry['confidence']}% confidence")
        st.caption(entry['text'][:100] + '...' if len(entry['text']) > 100 else entry['text'])
        st.divider()

    