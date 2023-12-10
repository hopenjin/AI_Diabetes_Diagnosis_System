from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
# from language_model import language_model
from pydantic import BaseModel
import json
import requests
url = 'http://localhost:5000/language-model'


app = FastAPI(title="Diabetes Prediction")

origins = [
    "http://0.0.0.0:8080",
    "http://localhost:8080",
    "http://0.0.0.0:2020",
    "http://localhost:2020",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.route("/process-text", methods=["OPTIONS"])
async def options_process_text():
    return {"methods": ["POST"]}

@app.post("/process-text")
async def process_text(requestData: TextRequest):
    text = requestData.text
    llm_response = requests.post(url, json={'text': text})
    data = json.loads(llm_response.text)
    response = data['text']
    # response = llm_response.text
    # response = text + " - processed"
    return response