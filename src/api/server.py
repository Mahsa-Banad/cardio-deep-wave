import os
import glob
import random

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.models.cnn_lstm import CNNLSTM
from src.data_loader.pipeline import get_available_records, process_record

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM()
model.load_state_dict(torch.load("saved_models/best_model.pth", map_location=device))
model.to(device)
model.eval()


CLASS_NAMES = ["N", "A", "V"]

class ECGInput(BaseModel):
    signal: list[float]

@app.get("/")
def root():
    return {"message" : "Cardioo Deep Wave API"}

@app.get("/health")
def health():
    return{"status" : "ok"}

@app.post("/predict")
def predict(data : ECGInput):
    if len(data.signal) != 180:
        return {"error" : f"Expected 180 samples, got {len(data.signal)}"}
    
    x = torch.tensor(data.signal, dtype = torch.float32).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return {"prediction" : CLASS_NAMES[prediction.item()],
            "confidence": round(confidence.item(), 4)}


@app.get("/predict/random")
def predict_random():
    records = list(get_available_records())

    if not records:
        return{"error" : "No record found in data/mit-bih"}
    
    random.shuffle(records)
    for record_id in records:
        try:
            beats, mapped_labels = process_record(record_id)
        except Exception:
            continue


        if len(beats)==0:
            continue

    
        idx = random.randint(0, len(beats) - 1)
        beat = beats[idx]
        true_label = mapped_labels[idx]

        x = torch.tensor(beat, dtype = torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
        return {
            "record" : record_id,
            "true_label" : CLASS_NAMES[true_label],
            "prediction" : CLASS_NAMES[prediction.item()],
            "confidence" : round(confidence.item(), 4),
          "correct" : CLASS_NAMES[true_label] == CLASS_NAMES[prediction.item()] 
            }
    return {"error" : "could not find any valid beats in any record."}