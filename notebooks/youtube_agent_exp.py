import textwrap
from dotenv import find_dotenv, load_dotenv

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# Load environment variables
load_dotenv(find_dotenv())

# Initialize embedings
emb_model_name = "sentence-transformers/all-mpnet-base-v2"
emb_model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model_name,
    model_kwargs=emb_model_kwargs,
    encode_kwargs=encode_kwargs,
)


def create_db_from_youtube_video_url(video_url):
    """The function embeds unstructured data and store the resulting embedding vectors
    in a Vector Sotore databse for further similarity verctor search.

    Args:
        video_url (url): A link to the YouTube video

    Returns:
        VectorStore: Return vector store database initialized from documents and embeddings
    """

    # Load YouTube transcripts
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # Split the transcript into chunks, embed each chunk and load it into the vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db


def get_response_from_query(vector_db, query, k=4):
    """
    LLMs can handle limited number of tokens (1024, 2048, 4096 and so on).
    Setting the chunksize and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = vector_db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Initialize the LLM model
    repo_id = "google/flan-t5-xxl"  # https://huggingface.co/google/flan-t5-xxl
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.6})  # type: ignore

    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching for the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    prompt = PromptTemplate(input_variables=["question", "docs"], template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs, docs_page_content


# Example usage:
video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
vector_db = create_db_from_youtube_video_url(video_url)

query = "What are they saying about Microsoft?"
response, docs, docs_page_content = get_response_from_query(vector_db, query)
print(textwrap.fill(response, width=85))
print(textwrap.fill(docs_page_content, width=85))
