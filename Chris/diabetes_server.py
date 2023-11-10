from fastapi import FastAPI, Form
# from language_model import language_model

app = FastAPI(title="Diabetes Prediction")

@app.post("/process-text")
async def process_text(text: str = Form("default test request")):
    # response = language_model(text)
    response = "dummy text"
    return {"response": response}