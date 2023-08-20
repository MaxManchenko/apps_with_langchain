import json
import textwrap
from dotenv import find_dotenv, load_dotenv

from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


class YouTubeAgent:
    """_summary_"""

    def __init__(self, video_url, query, config_path, k=3, fetch_k=5):
        """_summary_

        Args:
            video_url (url):
                _description_
            query (_type_):
                _description_
            config_path (str):
                _description_
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
        load_dotenv(find_dotenv())  # Load environment variables

    def load_config(self):
        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)
            self.llmodel = self.config.get("llm_hf")
            self.embedding = self.config.get("embeddings_hf")
            self.db = self.config.get("vector_db")

    def run(self):
        self.initialize_embeddings()
        self.initialize_llm()
        self.initialize_text_splitter()
        self.get_youtube_transcripts()
        self.create_vector_db()
        self.run_similarity_search()
        self.get_resopnse_from_llm()

    def initialize_embeddings(self):
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
        self.llm = HuggingFaceHub(
            repo_id=self.llmodel.get("repo_id"),
            model_kwargs=self.llmodel.get("model_kwargs", {"temperature": 0.8}),
        )  # type: ignore

    def initialize_text_splitter(self):
        """Split the transcript into chunks, embed each chunk for
        further loading to the vector store
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.db.get("chunk_size", 800),
            chunk_overlap=self.db.get("chunk_overlap", 100),
        )

    def get_youtube_transcripts(self):
        loader = YoutubeLoader.from_youtube_url(self.url)
        self.transcript = loader.load()

    def create_vector_db(self):
        """The function embeds unstructured data and store the resulting embedding vectors
        in a Vector Sotore databse for further similarity verctor search.
        (Vector Store = embedding vector + original splits)

        Returns:
            VectorStore: Return vector store database initialized from documents and embeddings
        """
        docs = self.text_splitter.split_documents(self.transcript)
        self.vector_db = FAISS.from_documents(docs, self.embeddings)

    def run_similarity_search(self):
        """LLMs can handle limited number of tokens (1024, 2048, 4096 and so on).
        That's why we're looking for snippets of text that are more relevant to the query.
        """
        docs = self.vector_db.max_marginal_relevance_search(
            self.query, self.k, self.fetch_k
        )
        self.docs_page_content = " ".join([d.page_content for d in docs])

    def get_resopnse_from_llm(self):
        """ """
        template = """
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript.
            
            Answer the following question: {question}
            By searching for the following video transcript: {document}
            
            Only use the factual information from the transcript to answer the question.
            
            If you feel like you don't have enough information to answer the question, say "I don't know".
            
            Your answers should be verbose and detailed.
            """
        prompt = PromptTemplate(
            input_variables=["question", "document"], template=template
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(question=self.query, docs=self.docs_page_content)
        response = response.replace("\n", "")

        return response


# Example usage:
