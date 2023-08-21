import json
import logging
import time
from dotenv import find_dotenv, load_dotenv

from langchain import HuggingFaceHub, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set the logging level
logging.basicConfig(level=logging.INFO)


class YouTubeAgent:
    """_summary_"""

    load_dotenv(find_dotenv())  # Load environment variables

    def __init__(self, video_url, query, config_path, *, k=3, fetch_k=5):
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
        self.query = query
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

    def run(self, max_retries=3):
        logging.info("Starting the YouTubeAgent...")
        self.initialize_embeddings()
        self.initialize_llm()
        self.initialize_memory()
        self.initialize_custom_template()
        self.initialize_text_splitter()
        self.get_youtube_transcripts()
        self.create_vector_db()
        self.initialize_retriever()
        self.initialize_qa_chain()

        # Retry loop
        for retry in range(max_retries):
            try:
                result = self.run_query()
                logging.info("YouTubeAgent run completed.")
                return result  # Exit loop if successful

            except ConnectionError as e:
                logging.warning(f"ConnectionError occurred: {e}")
                if retry < max_retries - 1:
                    logging.info("Retrying the run_query method after a short delay...")
                    time.sleep(3)  # Add a delay before retrying

        logging.error(f"YouTubeAgent run failed after {max_retries} retries.")
        return None

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
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            condense_question_prompt=self.CUSTOM_QUESTION_PROMPT,
            chain_type="stuff",
            memory=self.memory,
        )

    def run_query(self):
        """Get an answer to a question.

        Returns:
            dict: A dictionary with the question, chat_history and answer.
        """
        result = self.qa({"question": self.query})
        return result


# Example usage:
config_path = "../../configs/youtube.json"
video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
query = "What are they saying about Microsoft?"
yta = YouTubeAgent(video_url=video_url, query=query, config_path=config_path)
result = yta.run()
type(result)
print(result)
