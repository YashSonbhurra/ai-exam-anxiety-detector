import streamlit as st
import requests

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.anxiety-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    font-size: 18px;
}
.low {
    background-color: #e8f5e9;
    color: #1b5e20;
}
.moderate {
    background-color: #fff8e1;
    color: #e65100;
}
.high {
    background-color: #ffebee;
    color: #b71c1c;
}
.tip-box {
    background-color: #2589BD;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("🧠 AI-Based Exam Anxiety Detector")
st.write("Analyze exam-related thoughts and understand your anxiety level.")

# ---------- Input ----------
text = st.text_area(
    "✍️ Enter your thoughts before exam:",
    height=150,
    placeholder="Example: I am worried about my exam and can’t focus properly..."
)

# ---------- Button ----------
if st.button("🔍 Analyze Anxiety"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": text}
        )

        if response.status_code == 200:
            result = response.json()
            level = result["anxiety_level"]

            # ---------- Emoji + Message ----------
            if level == "Low Anxiety":
                st.markdown(
                    "<div class='anxiety-box low'>😌 <b>Low Anxiety</b><br>You seem calm and confident.</div>",
                    unsafe_allow_html=True
                )
                tips = [
                    "✔ Maintain your study routine",
                    "✔ Take short breaks",
                    "✔ Stay hydrated and relaxed"
                ]

            elif level == "Moderate Anxiety":
                st.markdown(
                    "<div class='anxiety-box moderate'>😟 <b>Moderate Anxiety</b><br>You seem a bit stressed.</div>",
                    unsafe_allow_html=True
                )
                tips = [
                    "📝 Revise key topics only",
                    "🧘 Try deep breathing exercises",
                    "⏳ Practice time management"
                ]

            else:  # High Anxiety
                st.markdown(
                    "<div class='anxiety-box high'>😰 <b>High Anxiety</b><br>You seem highly anxious.</div>",
                    unsafe_allow_html=True
                )
                tips = [
                    "🌬 Take slow deep breaths",
                    "📵 Avoid negative thoughts and distractions",
                    "🤝 Talk to a friend, teacher, or mentor",
                    "🛌 Get proper rest before exam"
                ]

            # ---------- Tips Section ----------
            st.subheader("💡 Anxiety Management Tips")
            for tip in tips:
                st.markdown(
                    f"<div class='tip-box'>{tip}</div>",
                    unsafe_allow_html=True
                )

        else:
            st.error("❌ Backend is not responding. Please check FastAPI server.")
