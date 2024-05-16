from langchain_community.document_loaders import ConfluenceLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from typing import Tuple, List

from constants import *
from utils import pretty_print_docs

class ConfluenceQA:
    def __init__(self):
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None
        self.retrieval_chain = None

        self.prompt = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to 
            answer the question. If you don't know the answer, just say that you don't know.
                                                     
            <context>
            {context}
            </context>
            
            Question: {input}
            
            """,
            role="system",
        )
        
        self.retriever = None
        self.retrieval_chain = None

    def init_embeddings(self) -> None:
        # OpenAI ada embeddings API
        self.embedding = OpenAIEmbeddings()

        self.vectordb = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=self.embedding,
            collection_name=DB_COLLECTION_NAME,
        )

    def init_models(self) -> None:
        # OpenAI GPT
        self.llm = ChatOpenAI(model_name=LLM, temperature=0.0)
        
        # Use local LLM hosted by LM Studio
        # self.llm = ChatOpenAI(
        #     openai_api_key = "NULL",
        #     temperature = 0,
        #     openai_api_base = "http://localhost:1234/v1"
        # )

    # TODO implement purge_data
    # def purge_data(self) -> None:
    #     """
    #     Clears the DB.
    #     """
    #     print("Clearing the DB. Collections before purge...")
    #     self.persistent_client.list_collections()
    #     self.persistent_client.delete_collection(DB_COLLECTION_NAME)
    #     print("Collections after purge...")
    #     self.persistent_client.list_collections()
    #     print("Cleared the DB.")

    def vector_db_confluence_docs(self, config: dict = {}) -> None:
        """
        Extracts documents from Confluence, splits them into chunks, and adds them to the database.

        Args:
            config (dict): A dictionary containing the configuration parameters.
                - confluence_url (str): The URL of the Confluence instance.
                - username (str): The username for authentication.
                - api_key (str): The API key for authentication.
                - space_key (str): The key of the Confluence space.
                - page_id (str): The ID of the Confluence page. If None, all pages in the space will be loaded.

        Returns:
            None
        """
        confluence_url = config.get("confluence_url", None)
        username = config.get("username", None)
        api_key = config.get("api_key", None)
        space_key = config.get("space_key", None)
        page_id = config.get("page_id", None)

        ## 1. Extract the documents
        loader = ConfluenceLoader(
            url=confluence_url, username=username, api_key=api_key
        )

        if page_id and page_id != "None":
            documents = loader.load(
                space_key=space_key, page_ids=[page_id], max_pages=1
            )
        else:
            documents = loader.load(space_key=space_key, limit=100, max_pages=1000)

        ## 2. Split the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=300
        )
        texts = text_splitter.split_documents(documents)

        ## 3. Add the documents to the DB
        if self.vectordb:
            print("count before", self.vectordb._collection.count())
            self.vectordb.add_documents(documents=texts)
            self.vectordb.persist()
            print("count after", self.vectordb._collection.count())
        else:
            # this should never happen?
            print("DB not initialized. Creating new DB from docs...")
            self.vectordb = Chroma.from_documents(
                documents=texts,
                embedding=self.embedding,
                persist_directory=DB_DIRECTORY,
            )
            self.persistent_client = self.vectordb.PersistentClient()
            print("count after", self.vectordb._collection.count())

    def retrieval_qa_chain(self) -> None:
        """
        Retrieves a question-answer chain using a custom prompt.

        Returns:
            None
        """
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.prompt)

        # TODO make threshold configurable
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.7, "k": 5},
        )
                
        compressor = FlashrankRerank()
        reranked_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
            )
    
        self.retrieval_chain = create_retrieval_chain(reranked_retriever, document_chain)

    def answer_confluence(self, question: str) -> Tuple[str, List[str]]:
        # Your code here
        """
        Answers a question using the Confluence QA system.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer to the question.
        """
        response = self.retrieval_chain.invoke({"input": question})

        answer = response["answer"]

        # we don't want duplicates in sources
        sources = set()
        
        compressed_docs = self.retriever.get_relevant_documents(question)
        for doc in compressed_docs:
            sources.add(doc.metadata["source"])
        
        # if using FlashrankRerank, the doc metata is not available in the response
        # for doc in response["context"]:
        #     sources.add(doc.metadata["source"])

        print(f"Number of sources: {len(sources)}")
        for source in sources:
            print(f"Sources: {source}")

        if not answer:
            print("LLM could not provide any answer.")
            answer = "Sorry, it seems I lack the domain information to answer that question. Try adding the data and ask again."

        return answer, sources
