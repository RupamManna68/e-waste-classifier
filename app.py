from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import requests
import shutil
import os

app = FastAPI()

# Serve the HTML file
app.mount("/static", StaticFiles(directory="static"), name="static")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

PROJECT = "balanced-e-waste-dataset"  # Replace with your project slug
VERSION = "3"  # Replace with your model version number
MODEL_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not ROBOFLOW_API_KEY:
        return {"error": "ROBOFLOW_API_KEY not set in environment variables."}

    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with open(temp_file_path, "rb") as img:
        resp = requests.post(
            MODEL_URL,
            params={"api_key": ROBOFLOW_API_KEY},
            files={"file": img},
        )

    return resp.json()
