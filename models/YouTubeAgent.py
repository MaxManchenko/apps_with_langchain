import json
import logging
import os
from dotenv import find_dotenv, load_dotenv
from typing import Optional

from langchain import HuggingFaceHub
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory

"""The key idea here is to decouple session-level initialization from per-video initialization.
You can initialize things like LLM, embeddings, and so on, at the session level,
and use that to initialize a YouTubeAgent whenever you get a new URL.

Here's how the classes are structured:
1) Initialize an instance of Initializer at the start of the user session.
This will set up everything that doesn't change across URLs.
2) Whenever a new URL comes up, a new YouTubeAgent is created using that instance of Initializer.
"""


class Initializer:
    """Initializer takes care of the static initialization tasks.
    Initialize the conversation bot components that do not change during the entire QA session.
    """

    load_dotenv(find_dotenv())
    logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.ERROR)

    def __init__(self, config_path: Optional[str] = None, k: int = 3, fetch_k: int = 5):
        """
        Args:
            config_path (str: optional):
                A path to the configuration file. Defaults to None.
            k (int: optional):
                The number of most relevant documents we get in search process.
                Defaults to 3.
            fetch_k (int: optional):
                The parameter sets how many documents you want to fetch before filtering in
                "max_marginal_relevance_search". In contrats to "k", the "fetch_k" parameter
                sets the number of both relevant and diverse documents we fetch.
                Defaults to 5.
        """
        self.k = k
        self.fetch_k = fetch_k
        self.config_path = config_path or os.getenv("CONFIG_PATH")
        self.load_config()
        self.initialize_embeddings()
        self.initialize_llm()
        self.initialize_memory()
        self.initialize_custom_template()
        self.initialize_text_splitter()

    def load_config(self):
        """Load data configuration from JSON file."""
        try:
            with open(self.config_path, "r") as config_file:
                self.config = json.load(config_file)
                logging.info(f"Loaded configuration from {self.config_path}")
                self.llmodel = self.config.get("llm_hf")
                self.embedding = self.config.get("embeddings_hf")
                self.db = self.config.get("vector_db")
                logging.info("The instance is built")

        except FileNotFoundError:
            logging.error(f"Data config file at {self.config_path} not found.")
            raise

    def initialize_embeddings(self):
        """Initialize embeddings to convert data to vectors."""
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
        """Initialize an LLM for QA chain."""
        self.llm = HuggingFaceHub(
            repo_id=self.llmodel.get("repo_id"),
            model_kwargs=self.llmodel.get("model_kwargs", {"temperature": 0.1}),
        )  # type: ignore
        logging.info(f"The LLM {self.llm} has been created")

    def initialize_memory(self):
        """Initialize memory for QA chain."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )  # type: ignore

    def initialize_custom_template(self):
        """Initialize custom template for QA chain."""
        custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the end of standalone question add this 'Answer the question in German language.' If you do not know the answer reply with 'I am sorry'."""
        self.CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    def initialize_text_splitter(self):
        """Split the transcript into chunks, embed each chunk for
        further loading to the vector store.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.db.get("chunk_size", 800),
            chunk_overlap=self.db.get("chunk_overlap", 100),
        )  # type: ignore


class YouTubeAgent:
    """YouTubeAgent handles the dynamic aspects tied to a specific YouTube URL.
    Initialize the conversational bot components and create data
    that change when a new URL comes up."""

    def __init__(self, video_url, initializer: Initializer):
        """
        Args:
            video_url (URL): A YuTube video URL
            initializer (Initializer): Handles static aspects of the user session.
        """
        self.url = video_url
        self.vector_db = None
        self.initializer = initializer

        # Use attributes from the shared Initializer
        self.embeddings = initializer.embeddings
        self.llm = initializer.llm
        self.memory = initializer.memory
        self.memory.clear()
        self.text_splitter = initializer.text_splitter
        self.CUSTOM_QUESTION_PROMPT = initializer.CUSTOM_QUESTION_PROMPT
        self.k = initializer.k
        self.fetch_k = initializer.fetch_k

    def run(self):
        """Create an agent ready to go.

        Returns:
            ConversationalRetrievalChain instance
        """
        self.get_youtube_transcripts()
        self.get_or_create_vector_db()
        self.initialize_retriever()
        return self.initialize_qa_chain()

    def get_youtube_transcripts(self):
        """Loading documents from a YouTube url."""
        loader = YoutubeLoader.from_youtube_url(self.url)
        self.transcript = loader.load()

    def get_or_create_vector_db(self):
        """The function embeds unstructured data and store the resulting embedding vectors
        in a Vector Sotore databse for further verctor search.
        (Vector Store = embedding vector + original splits).

        Returns:
            VectorStore: Return vector store database initialized from documents and embeddings.
        """
        if self.vector_db is None:
            logging.info("Starting creating the vector base...")
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
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=self.CUSTOM_QUESTION_PROMPT,
            chain_type="stuff",
            memory=self.memory,
        )
        logging.info("The YouTubeAgent is ready to go.")
        return qa_chain
