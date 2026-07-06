# utils/inference.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.preprocessing import preprocess_text

# =========================
# LABEL MAP
# =========================
LABEL_MAP = {
    0: "Positif",
    1: "Negatif",
    2: "Netral"
}

# =========================
# DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD MODEL FUNCTION
# =========================
def load_sentiment_model(model_path: str):
    """
    Load tokenizer + model sekali saja
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return tokenizer, model


# =========================
# PREDICTION FUNCTION
# =========================
def predict_sentiment(text: str, tokenizer, model):
    """
    Inferensi sentimen 1 kalimat
    """

    # preprocessing
    clean_text = preprocess_text(text)

    # tokenize
    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()
    confidence = float(probs[0][pred_id])

    return {
        "text_original": text,
        "text_clean": clean_text,
        "label": LABEL_MAP[pred_id],
        "confidence": confidence,
        "probabilities": probs.cpu().numpy()[0].tolist()
    }


# =========================
# BATCH INFERENCE
# =========================
def predict_batch(texts, tokenizer, model):
    """
    Untuk CSV / dataset besar
    """
    results = []

    for text in texts:
        results.append(predict_sentiment(text, tokenizer, model))

    return results