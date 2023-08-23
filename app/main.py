from fastapi import FastAPI
from pydantic import BaseModel

from app.model import __version__ as model_version
from app.model import run_youtube_agent


app = FastAPI()


class TextIn(BaseModel):
    video_url: str
    question: str


class PredictionOut(BaseModel):
    answer: dict


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "model_version"}


@app.post("/get_answer", response_model=PredictionOut)
def get_answer(payload: TextIn):
    answer = run_youtube_agent(payload.video_url, payload.question)
    return answer
