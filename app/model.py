from models.YouTubeAgent import Initializer, YouTubeAgent
from src.utils.query_runner import run_query


__version__ = "0.2.0"


def run_youtube_agent(video_url, query):
    """_summary_

    Args:
        video_url (URL): a YouTube video URL
        query (str): request

    Returns:
        (YouTubeAgent instance, answer)
    """
    initializer = Initializer()
    yta = YouTubeAgent(video_url=video_url, initializer=initializer)
    agent = yta.run()
    answer = run_query(agent, query)
    return agent, answer
