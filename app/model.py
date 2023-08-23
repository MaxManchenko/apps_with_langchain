from models.YouTubeAgent import YouTubeAgent
from src.utils.query_runner import run_query

config_path = "configs/youtube.json"

__version__ = "0.1.0"

def run_youtube_agent(video_url, query):
    yta = YouTubeAgent(video_url=video_url, config_path=config_path)
    agent = yta.build_agent()
    answer = run_query(agent, query)
    return answer
