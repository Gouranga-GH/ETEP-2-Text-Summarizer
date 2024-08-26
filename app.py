# Import the libraries
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from starlette.responses import RedirectResponse
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()

# Set up the templates directory
templates = Jinja2Templates(directory="templates")

# Set up the static directory for custom CSS, JS, etc.
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["authentication"], response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train", response_class=HTMLResponse)
async def training(request: Request):
    try:
        os.system("python main.py")
        return templates.TemplateResponse("train.html", {"request": request, "message": "Training successful!"})
    except Exception as e:
        return templates.TemplateResponse("train.html", {"request": request, "message": f"Error Occurred! {e}"})

@app.get("/predict", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, text: str = Form(...)):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return templates.TemplateResponse("predict.html", {"request": request, "summary": summary, "input_text": text})
    except Exception as e:
        return templates.TemplateResponse("predict.html", {"request": request, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
