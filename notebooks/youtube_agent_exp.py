import textwrap
from dotenv import find_dotenv, load_dotenv

from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory


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


def get_response_from_query(vector_db, query, k=3):
    """
    LLMs can handle limited number of tokens (1024, 2048, 4096 and so on).
    Setting the chunksize and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = vector_db.max_marginal_relevance_search(query, k=k, fetch_k=5)
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

# ------------------------------------------------------
# Use retriever for similarity rearch
# ------------------------------------------------------


def get_response_from_query_2(vector_db, query):
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
    )

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

    response = chain.run(question=query, docs=retriever)
    response = response.replace("\n", "")
    return response, retriever


# Example usage:
query = "What are they saying about Microsoft?"
response, retriever = get_response_from_query_2(vector_db, query)
print(textwrap.fill(response, width=85))


# ------------------------------------------------------
# Using a custom prompt for condensing the question
# ------------------------------------------------------
video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in German language.' If you do not know the answer reply with 'I am sorry'."""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


# Initialize the LLM model
repo_id = "google/flan-t5-xxl"  # https://huggingface.co/google/flan-t5-xxl
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.0})  # type: ignore

# Initialize embedings
emb_model_name = "sentence-transformers/all-mpnet-base-v2"
emb_model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model_name,
    model_kwargs=emb_model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Initialize loader
loader = YoutubeLoader.from_youtube_url(video_url)
transcript = loader.load()

# Split the transcript into chunks, embed each chunk and load it into the vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(transcript)
vector_db = FAISS.from_documents(docs, embeddings)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the LLM model
repo_id = "google/flan-t5-xxl"  # https://huggingface.co/google/flan-t5-xxl
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.6})  # type: ignore

# Initialize retriever
retriever = vector_db.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    chain_type="stuff",
    memory=memory,
)

# Example usage:
query = "What are they saying about Microsoft?"
result = qa({"question": query})
