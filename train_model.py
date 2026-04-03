import os
from pyexpat import model
print("Current folder:", os.getcwd())
print("Files in folder:", os.listdir())
print("Running train_model.py...")
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import nltk

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("spam.csv", names=["label","message"], encoding="latin-1").dropna()
df['label'] = df['label'].map({"ham":0,"spam":1})

# Preprocess function
def preprocess(text):
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['message'] = df['message'].apply(preprocess)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train model
import os
# Save model & vectorizer with full path
joblib.dump(model, os.path.join(os.getcwd(), "spam_model.pkl"))

joblib.dump(vectorizer, os.path.join(os.getcwd(), "tfidf_vectorizer.pkl"))
print("✅ Model & Vectorizer saved in", os.getcwd())
