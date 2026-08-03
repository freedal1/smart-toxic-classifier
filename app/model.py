import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Путь к модели (от корня проекта)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rubert-toxic")
MAX_LENGTH = 128

# Глобальные переменные для модели и токенизатора (загружаются один раз)
_model = None
_tokenizer = None
_device = None


def load_model():
    """Загружает модель и токенизатор (ленивая загрузка)"""
    global _model, _tokenizer, _device

    if _model is not None:
        return  # уже загружено

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _model.to(_device)
    _model.eval()
    print(f"✅ Модель загружена на устройство: {_device}")


def predict(text: str):
    """
    Предсказание токсичности текста.
    Возвращает словарь: text, toxic (bool), confidence (float)
    """
    load_model()  # гарантируем, что модель загружена

    if not text or not isinstance(text, str) or not text.strip():
        return {
            "text": text,
            "toxic": False,
            "confidence": 0.0,
            "error": "Empty or invalid text provided"
        }

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)
    toxic_probability = probabilities[0][1].item()

    return {
        "text": text,
        "toxic": toxic_probability > 0.5,
        "confidence": round(toxic_probability, 3)
    }