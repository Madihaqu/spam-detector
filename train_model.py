import os
import pandas as pd
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Current folder:", os.getcwd())
print("Files in folder:", os.listdir())
print("Running train_model.py...")

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset — skip header row, use only first 2 columns
df = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0, 1], header=0)
df.columns = ["label", "message"]
df.dropna(inplace=True)

# Convert labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df.dropna(subset=["label"], inplace=True)  # drop rows where map failed

# Preprocess function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return " ".join(words)
    return ""

df["message"] = df["message"].apply(preprocess)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)
print("✅ Model trained successfully")

# Save model & vectorizer
joblib.dump(model, os.path.join(os.getcwd(), "spam_model.pkl"))
joblib.dump(vectorizer, os.path.join(os.getcwd(), "tfidf_vectorizer.pkl"))
print("✅ Model & Vectorizer saved in", os.getcwd())