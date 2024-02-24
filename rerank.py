from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from dotenv import load_dotenv

load_dotenv()

# import pretty print
from utils import pretty_print_docs

from constants import *


prompt = ChatPromptTemplate.from_template(
            """Answer the following question based only on the provided context:
                                                     
            <context>
            {context}
            </context>
            
            Question: {input}
            
            """,
            role="system",
        )

# OpenAI ada embeddings API
embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=DB_DIRECTORY,
    embedding_function=embedding,
    collection_name=DB_COLLECTION_NAME,
)

llm = ChatOpenAI(model_name=LLM, temperature=0.0)

document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

standard_retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 5},
        )

retrieval_chain = create_retrieval_chain(standard_retriever, document_chain)

response = retrieval_chain.invoke({"input": "What are the goals of the AI RAG app POC?"})

print(f"\n ***** standard response: {response['answer']}")

# docs = standard_retriever.get_relevant_documents(query)
# print("\n ***** original docs:")
# pretty_print_docs(docs)

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=standard_retriever
)

# print("\n ****** compressed docs:")
# compressed_docs = compression_retriever.get_relevant_documents(
#     "What are the goals of the AI RAG app POC?"
# )
# pretty_print_docs(compressed_docs)


retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)

response = retrieval_chain.invoke({"input": "What are the goals of the AI RAG app POC?"})

print(f"\n ***** reranked response: {response['answer']}")