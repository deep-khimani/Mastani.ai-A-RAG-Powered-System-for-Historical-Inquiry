import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma 
from langchain.chains import RetrievalQA

load_dotenv()
DB_PATH = "pune_vector_db"

def create_qa_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return chain

def get_ai_response(question: str):
    qa_chain = create_qa_chain()
    result = qa_chain.invoke({"query": question})
    return result["result"]