import logging

from fastapi import FastAPI, Cookie
from pydantic import BaseModel
import uuid

from src.utils.query_runner import run_query
from app.model import __version__ as model_version
from app.model import run_youtube_agent


app = FastAPI()
logging.basicConfig(level=logging.INFO)


class Query(BaseModel):
    video_url: str = None
    question: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


agents = {}  # In-memory data store mapping session IDs to YouTubeAgent instances


@app.post("/get_answer")
async def get_answer(query: Query, interaction_id: str = Cookie(None)):
    if interaction_id is None:  # Generate a new interaction ID if not provided
        interaction_id = str(uuid.uuid4())

    # If there's a new video URL, create a new YouTubeAgent instance
    if query.video_url:
        agent, answer = run_youtube_agent(
            video_url=query.video_url, query=query.question
        )
        agents[interaction_id] = agent  # Store the new YouTubeAgent instance
        return {"answer": answer, "interaction_id": interaction_id}

    # Use existing YouTubeAgent for this interaction_id if it exists
    existing_agent = agents.get(interaction_id)

    if existing_agent is None:
        return {
            "error": "No video URL specified for this interaction_id",
            "interaction_id": interaction_id,
        }

    # Otherwise, use the existing YouTubeAgent to answer the query
    try:
        answer = run_query(existing_agent, query.question)
        return {"answer": answer, "interaction_id": interaction_id}
    except Exception as e:
        logging.warning(f"Query failed with error: {e}")
        return {
            "error": f"Query failed with error: {e}",
            "interaction_id": interaction_id,
        }
