from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def run_query(qa_instance, query):
    return qa_instance({"question": query})
