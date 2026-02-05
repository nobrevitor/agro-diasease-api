from fastapi import FastAPI, UploadFile, File
from app.model_loader import load_model
from app.preprocessing import preprocess_image
import torch.nn.functional as F
import torch

app = FastAPI(title="Agro Diesease Detection API")

model_milho = load_model("models/milho/model.pt")
model_soja = load_model("models/soja/model.pt")

@app.post("/predict/milho")
async def predict_milho(file: UploadFile = File(...)):
    image = preprocess_image(file.file)

    with torch.no_grad():
        output = model_milho(image)
        probs = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return {
        "cultura": "milho",
        "classe": prediction.item(),
        "confianca": round(confidence.item() * 100, 2)
    }

@app.post("/predict/soja")
async def predict_soja(file: UploadFile = File(...)):
    image = preprocess_image(file.file)

    with torch.no_grad():
        output = model_soja(image)
        probs = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return {
        "cultura": "soja",
        "classe": prediction.item(),
        "confianca": round(confidence.item() * 100, 2)
    }