import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, HTTPException
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ----- ПУТЬ К МОДЕЛИ (гарантированно работает) -----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rubert-toxic")
MAX_LENGTH = 128

# ----- ИМПОРТ СХЕМ -----
from schemas import TextRequest, PredictionResponse

# ----- ЗАГРУЗКА МОДЕЛИ -----
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    print(f"✅ Модель загружена с {MODEL_PATH} на устройство: {device}")

except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    raise RuntimeError("Не удалось загрузить модель.")


# ----- ФУНКЦИЯ ПРЕДСКАЗАНИЯ -----
def predict(text: str):
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "text": text,
            "toxic": False,
            "confidence": 0.0,
            "error": "Empty or invalid text provided"
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    toxic_probability = probabilities[0][1].item()

    return {
        "text": text,
        "toxic": toxic_probability > 0.5,
        "confidence": round(toxic_probability, 3)
    }


# ----- FASTAPI -----
app = FastAPI(
    title="Smart Toxic Classifier API",
    description="Классификация комментариев на токсичные и нетоксичные",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Smart Toxic Classifier API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.post("/predict")
def predict_endpoint(request: TextRequest):
    try:
        result = predict(request.text)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))