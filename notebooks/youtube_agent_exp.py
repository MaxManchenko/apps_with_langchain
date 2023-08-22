import json
import logging
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

import sys

sys.path.insert(0, "../../apps_with_langchain")
from src.utils.query_runner import run_query


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


# ------------------------------------------------------
# YouTubeAgent class
# ------------------------------------------------------


class YouTubeAgent:
    """_summary_"""

    load_dotenv(find_dotenv())  # Load environment variables
    logging.basicConfig(level=logging.INFO)

    def __init__(self, video_url, config_path, *, k=3, fetch_k=5):
        """_summary_

        Args:
            video_url (url):
                A YouTube url
            query (_type_):
                A question about the video
            config_path (str):
                A path to the configuration file
            k (int, optional):
                The number of most relevant documents we get in search process.
                Defaults to 3.
            fetch_k (int, optional):
                The parameter sets how many documents you want to fetch before filtering in
                "max_marginal_relevance_search". In contrats to "k", the "fetch_k" parameter
                sets the number of both relevant and diverse documents we fetch.
                Defaults to 5.
        """
        self.url = video_url
        self.k = k
        self.fetch_k = fetch_k
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)
            logging.info(f"Loaded configuration from {self.config_path}")
            self.llmodel = self.config.get("llm_hf")
            self.embedding = self.config.get("embeddings_hf")
            self.db = self.config.get("vector_db")
            logging.info("The instance is built")

    def build_agent(self):
        logging.info("Starting the YouTubeAgent...")
        self.initialize_embeddings()
        self.initialize_llm()
        self.initialize_memory()
        self.initialize_custom_template()
        self.initialize_text_splitter()
        self.get_youtube_transcripts()
        self.create_vector_db()
        self.initialize_retriever()
        return self.initialize_qa_chain()

    def initialize_embeddings(self):
        """Initialize embeddings to convert data to vectors"""
        embedder = self.embedding.get(
            "model_name", "sentence-transformers/all-mpnet-base-v2"
        )
        embedder_kwargs = self.embedding.get("model_kwargs", {"device": "cpu"})
        encode_kwargs = self.embedding.get(
            "encode_kwargs", {"normalize_embeddings": False}
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedder,
            model_kwargs=embedder_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def initialize_llm(self):
        """Initialize a LLM for QA chain"""
        self.llm = HuggingFaceHub(
            repo_id=self.llmodel.get("repo_id"),
            model_kwargs=self.llmodel.get("model_kwargs", {"temperature": 0.1}),
        )  # type: ignore
        logging.info(f"The LLM {self.llm} has been created")

    def initialize_memory(self):
        """Initialize memory for QA chain"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    def initialize_custom_template(self):
        """Initialize custom template for QA chain"""
        custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in German language.' If you do not know the answer reply with 'I am sorry'."""
        self.CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    def initialize_text_splitter(self):
        """Split the transcript into chunks, embed each chunk for
        further loading to the vector store
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.db.get("chunk_size", 800),
            chunk_overlap=self.db.get("chunk_overlap", 100),
        )

    def get_youtube_transcripts(self):
        """Loading documents from a YouTube url"""
        loader = YoutubeLoader.from_youtube_url(self.url)
        self.transcript = loader.load()

    def create_vector_db(self):
        logging.info("Starting creating the vector base...")
        """The function embeds unstructured data and store the resulting embedding vectors
        in a Vector Sotore databse for further verctor search.
        (Vector Store = embedding vector + original splits)

        Returns:
            VectorStore: Return vector store database initialized from documents and embeddings
        """
        docs = self.text_splitter.split_documents(self.transcript)
        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        logging.info("The vector base has been created")

    def initialize_retriever(self):
        """Look for snippets of text that are more relevant to the query."""
        self.retriever = self.vector_db.as_retriever(
            search_type="mmr", search_kwargs={"k": self.k, "fetch_k": self.fetch_k}
        )

    def initialize_qa_chain(self):
        """Initialize a ConversationalRetrievalQA chain with a chat history component."""
        qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=self.CUSTOM_QUESTION_PROMPT,
            chain_type="stuff",
            memory=self.memory,
        )
        logging.info("The YouTubeAgent is redy to go.")
        return qa


# Example usage:
config_path = "../configs/youtube.json"
video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
query = "What are they saying about Microsoft?"

yta = YouTubeAgent(video_url=video_url, config_path=config_path)
agent = yta.build_agent()
result = run_query(agent, query)


