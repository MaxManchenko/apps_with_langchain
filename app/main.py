import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.model import __version__ as model_version
from app.model import run_youtube_agent


app = FastAPI()
logging.basicConfig(level=logging.INFO)


class TextIn(BaseModel):
    video_url: str
    question: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/get_answer")
def get_answer(payload: TextIn):
    logging.info(f"Received payload: {payload.dict()}")
    answer = run_youtube_agent(payload.video_url, payload.question)
    logging.info(f"Returning answer: {answer}")
    return answer
