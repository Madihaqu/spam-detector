import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Download stopwords (only first time)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# 1. Load dataset — skip header, use only first 2 columns
df = pd.read_csv("spam.csv", encoding="latin-1", usecols=[0, 1], header=0)
df.columns = ["label", "message"]

# 2. Remove empty rows
df.dropna(inplace=True)

# 3. Convert labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})
df.dropna(subset=["label"], inplace=True)  # drop rows where map failed

# 4. Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return ""

df["message"] = df["message"].apply(preprocess)

# 5. Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Interactive user input
while True:
    msg = input("\nEnter message (type 'exit' to quit): ")
    if msg.lower() == "exit":
        break

    processed = preprocess(msg)
    msg_vector = vectorizer.transform([processed])
    result = model.predict(msg_vector)[0]

    if result == 1:
        print("Spam 🚫")
    else:
        print("Ham ✅")