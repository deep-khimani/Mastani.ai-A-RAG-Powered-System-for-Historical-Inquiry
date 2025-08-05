import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DB_PATH = "pune_vector_db"
DATA_PATH = "data"

def create_vector_db():
    print("--- Loading source documents ---")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    print("--- Splitting documents into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(text_chunks)} chunks.")

    print("--- Initializing Google AI embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Embeddings model initialized.")

    print("--- Creating and persisting vector store ---")
    
    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"--- Vector store created and saved to '{DB_PATH}' ---")
    print("--- Ingestion Complete! ---")


if __name__ == "__main__":
    create_vector_db()