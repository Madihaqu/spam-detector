from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = FastAPI(title="Spam Detector API")

# Load ML models
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

class MessageRequest(BaseModel):
    message: str

def preprocess_text(text: str) -> str:
    text = text.lower().split()
    text = [w for w in text if w not in stop_words]
    return " ".join(text)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict_spam(payload: MessageRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    cleaned_text = preprocess_text(payload.message)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = float(probabilities[prediction]) * 100

    return {
        "is_spam": bool(prediction == 1),
        "label": "SPAM" if prediction == 1 else "HAM",
        "confidence": round(confidence, 2)
    }