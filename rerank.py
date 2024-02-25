from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

from dotenv import load_dotenv

# import pretty print
from utils import pretty_print_docs
from constants import *

load_dotenv()

standard_retriever = None
standard_retrieval_chain = None
reranked_retriever = None
reranked_retrieval_chain = None

def setup():
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
    standard_retrieval_chain = create_retrieval_chain(standard_retriever, document_chain)
    
    compressor = FlashrankRerank()
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=standard_retriever
    )
    
    reranked_retrieval_chain = create_retrieval_chain(reranked_retriever, document_chain)
    
    return standard_retriever, standard_retrieval_chain, reranked_retriever, reranked_retrieval_chain

def invoke_standard_retrieval_chain(query, verbose=False):
    response = standard_retrieval_chain.invoke({"input": query})
    if verbose:
        print("\n ****** original docs:")
        standard_docs = standard_retriever.get_relevant_documents(query)
        pretty_print_docs(standard_docs)  
    return response

def invoke_reranked_retrieval_chain(query, verbose=False):
    response = reranked_retrieval_chain.invoke({"input": query})
    if verbose:
        print("\n ****** compressed docs:")
        compressed_docs = reranked_retriever.get_relevant_documents(query)
        pretty_print_docs(compressed_docs)  
    return response

standard_retriever, standard_retrieval_chain, reranked_retriever, reranked_retrieval_chain = setup()

def main():
    setup()
    user_input = ""
    while user_input != "x":
        user_input = input("Enter your query or 'x' to exit: ")
        if user_input != 'x':
            standard_response = invoke_standard_retrieval_chain(user_input, verbose=True)
            print(f"\n ***** standard_response: {standard_response['answer']}")
            reranked_response = invoke_reranked_retrieval_chain(user_input, verbose=True)
            print(f"\n ***** reranked_response: {reranked_response['answer']}")
        else:
            break
        
if __name__ == "__main__":
    main()