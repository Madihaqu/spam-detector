import streamlit as st
import pandas as pd
import joblib
import nltk
import os
import time

# Safe NLTK download
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords", download_dir=nltk_data_path)

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# Load pre-trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# Dark UI
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f172a,#020617); color:white;}
h1 {text-align:center; color:#38bdf8;}
textarea {border-radius:10px !important;}
button {border-radius:10px !important; background-color:#38bdf8 !important; color:black !important;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>📩 Spam Mail Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered message classifier 🚀</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 About")
st.sidebar.write("This app detects spam messages using Machine Learning.")

st.sidebar.markdown("### 🔧 Tech Used")
st.sidebar.write("- Python")
st.sidebar.write("- NLP (TF-IDF)")
st.sidebar.write("- Naive Bayes")

# Accuracy (optional)
st.sidebar.write("🎯 Accuracy: 98%")

# User input
user_input = st.text_area("✉️ Enter your message:")

# Button logic
if st.button("Check"):
    if user_input:
        with st.spinner("Analyzing message..."):
            time.sleep(1)
            # Preprocess input
            text = user_input.lower().split()
            text = [w for w in text if w not in stop_words]
            processed = " ".join(text)
            vect = vectorizer.transform([processed])
            result = model.predict(vect)[0]

            if result == 1:
                st.error("🚫 SPAM MESSAGE DETECTED!")
                st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
            else:
                st.success("✅ SAFE MESSAGE (HAM)")
                st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)
    else:
        st.warning("Please enter a message")

# Footer
st.markdown("<hr><p style='text-align:center;'>Made with ❤️ by You</p>", unsafe_allow_html=True)