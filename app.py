import streamlit as st
import pandas as pd
import nltk
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

# UI Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

h1 {
    text-align: center;
    color: #38bdf8;
}

textarea {
    border-radius: 10px !important;
}

button {
    border-radius: 10px !important;
    background-color: #38bdf8 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# Title (ONLY ONCE ✅)
st.markdown("<h1>📩 Spam Mail Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered message classifier 🚀</p>", unsafe_allow_html=True)

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('spam.csv', sep=',', names=['label', 'message'], encoding='latin-1')
df.dropna(inplace=True)

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return ""

df['message'] = df['message'].apply(preprocess)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Sidebar
st.sidebar.title("📊 About")
st.sidebar.write("This app detects spam messages using Machine Learning.")

st.sidebar.markdown("### 🔧 Tech Used")
st.sidebar.write("- Python")
st.sidebar.write("- NLP (TF-IDF)")
st.sidebar.write("- Naive Bayes")

# Accuracy
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
st.sidebar.write(f"🎯 Accuracy: {round(acc*100, 2)}%")

# User input
user_input = st.text_area("✉️ Enter your message:")

# Button logic (ONLY ONCE ✅)
if st.button("Check"):
    if user_input:
        with st.spinner("Analyzing message..."):
            time.sleep(1)

            processed = vectorizer.transform([user_input])
            result = model.predict(processed)[0]

            if result == 1:
                st.error("🚫 SPAM MESSAGE DETECTED!")
                st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=100)
            else:
                st.success("✅ SAFE MESSAGE (HAM)")
                st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)
    else:
        st.warning("Please enter a message")

# Footer
st.markdown("""
<hr>
<p style='text-align:center;'>Made with ❤️ by You</p>
""", unsafe_allow_html=True)