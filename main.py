import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords (only first time)
nltk.download('stopwords')

# 1. Load dataset
df = pd.read_csv('spam.csv', sep=',', names=['label', 'message'], encoding='latin-1')

# 2. Remove empty rows
df.dropna(inplace=True)

# 3. Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Stopwords
stop_words = set(stopwords.words('english'))

# 5. Safe preprocessing function
def preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return ""

# Apply preprocessing
df['message'] = df['message'].apply(preprocess)

# 6. Convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 9. Test model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# 10. User input
while True:
    msg = input("\nEnter message (type 'exit' to quit): ")
    if msg.lower() == "exit":
        break

    msg_vector = vectorizer.transform([msg])
    result = model.predict(msg_vector)[0]


    if result == 1:
        print("Spam 🚫")
    else:

        print("Ham ✅")
        from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()