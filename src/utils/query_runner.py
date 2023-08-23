from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def run_query(qa_instance, query):
    """Wrapper for additional 'retry' functionality:
    make it possibe to call an Langchain instance several times
    without waiting when an exception is raised.

    Args:
        qa_instance (int): A LangChain instance
        query (_type_): _description_

    Returns:
        dict: {"question" : "your question",
            "chat_hystory" : "list of chat history",
            "answer" : "An LLM answer to your question"}.
    """
    return qa_instance({"question": query})
