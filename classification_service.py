from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="Spam Classification Service", version="1.0")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Charger le modèle et le vectoriseur
model_dir = 'ml_models'
model = joblib.load(os.path.join(model_dir, 'logistic_regression_model.joblib'))
vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))

class EmailInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    text: str
    prediction: str
    confidence: float

def preprocess_email(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

@app.post("/predict", response_model=PredictionOutput)
def predict_spam(data: EmailInput):
    processed = preprocess_email(data.text)
    X = vectorizer.transform([processed])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    
    return PredictionOutput(
        text=data.text[:100],
        prediction=prediction,
        confidence=float(confidence)
    )

@app.post("/batch-predict")
def batch_predict(emails: list[EmailInput]):
    results = []
    for email in emails:
        processed = preprocess_email(email.text)
        X = vectorizer.transform([processed])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities)
        
        results.append({
            "text": email.text[:50],
            "prediction": prediction,
            "confidence": float(confidence)
        })
    return {"predictions": results}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Spam Classification Service", "model": "logistic_regression"}

@app.get("/info")
def service_info():
    return {
        "service": "Spam Classification Service",
        "version": "1.0",
        "model": "Logistic Regression",
        "vectorizer": "TF-IDF",
        "endpoints": [
            "/predict - POST: Predict if email is spam",
            "/batch-predict - POST: Predict multiple emails",
            "/health - GET: Service health check",
            "/info - GET: Service information"
        ]
    }
