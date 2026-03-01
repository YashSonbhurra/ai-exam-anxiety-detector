from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer
import sys
import os

# add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.model import BertAnxietyClassifier

app = FastAPI(title="AI Exam Anxiety Detector API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertAnxietyClassifier()
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "model", "bert_anxiety_model.pt"),
        map_location=DEVICE
    )
)
model.to(DEVICE)
model.eval()


class TextRequest(BaseModel):
    text: str


@app.post("/predict")
def predict_anxiety(request: TextRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        pred = torch.argmax(logits, dim=1).item()

    label_map = {
        0: "Low Anxiety",
        1: "Moderate Anxiety",
        2: "High Anxiety"
    }

    return {"anxiety_level": label_map[pred]}
