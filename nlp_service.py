from fastapi import FastAPI
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pydantic import BaseModel

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="NLP Preprocessing Service", version="1.0")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    original: str
    cleaned: str

@app.post("/clean", response_model=TextOutput)
def clean_text(data: TextInput):
    text = data.text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    cleaned = ' '.join(tokens)
    return TextOutput(original=data.text, cleaned=cleaned)

@app.post("/tokenize")
def tokenize(data: TextInput):
    text = data.text.lower()
    tokens = word_tokenize(text)
    return {"tokens": tokens, "count": len(tokens)}

@app.post("/stem")
def stem_text(data: TextInput):
    tokens = word_tokenize(data.text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    return {"original": tokens, "stemmed": stemmed}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "NLP Preprocessing Service"}
