import nest_asyncio
nest_asyncio.apply()

import gradio as gr
import time
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gradio.themes.base import Base
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration for Render
PORT = int(os.environ.get("PORT", 7860))
HOST = os.environ.get("HOST", "0.0.0.0")

DB_PATH = "pune_vector_db"

# Global variable to store the QA chain
qa_chain = None

# Load header HTML with error handling
def load_header_html():
    try:
        with open("templates/header.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.warning("Header HTML file not found, using default")
        return "<h1>🏛️ Mastani.ai - Historical Inquiry System</h1>"

header_html = load_header_html()

def load_documents():
    """Load documents from data directory"""
    documents = []
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found!")
        return documents
    
    try:
        # Load text files
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            if filename.endswith('.txt'):
                logger.info(f"Loading text file: {filename}")
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            
            elif filename.endswith('.pdf'):
                logger.info(f"Loading PDF file: {filename}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def create_vector_database():
    """Create vector database from documents"""
    global qa_chain
    
    try:
        # Check if database already exists
        if os.path.exists(DB_PATH):
            logger.info("Vector database already exists, loading...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            return vector_store
        
        logger.info("Creating new vector database...")
        
        # Load documents
        documents = load_documents()
        if not documents:
            logger.error("No documents found to create vector database")
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(texts)} chunks")
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        logger.info("Vector database created successfully!")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        return None

def create_qa_chain():
    """Create QA chain with error handling"""
    global qa_chain
    
    if qa_chain is not None:
        return qa_chain
    
    try:
        # Check if required environment variables are set
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
        
        # Create or load vector database
        vector_store = create_vector_database()
        if vector_store is None:
            return None
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        logger.info("QA chain created successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        return None

def get_ai_response(question: str):
    """Get AI response with error handling"""
    try:
        chain = create_qa_chain()
        if chain is None:
            return "I apologize, but the historical database is currently being set up. Please try again in a moment."
        
        result = chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return f"I encountered an error while processing your question. Please try rephrasing your query."

class MidnightDurbar(Base):
    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.pink,
            font=(gr.themes.GoogleFont("Lora"), "ui-sans-serif", "system-ui", "sans-serif"),
            font_mono=(gr.themes.GoogleFont("Roboto Mono"), "ui-monospace", "Consolas", "monospace"),
        )
        self.set(
            body_background_fill="#101010",
            background_fill_primary="#1c1c1c",
            body_text_color="#f0f0f0",
            body_text_color_subdued="#a0a0a0",
            slider_color="#E4007C",
            button_primary_background_fill="#E4007C",
            button_primary_text_color="white",
            button_primary_background_fill_hover="#C13584",
            background_fill_secondary="#E4007C",
            block_background_fill="#282828",
            border_color_accent="#333333",
            block_border_width="1px",
            block_shadow="*shadow_drop_lg",
        )

midnight_theme = MidnightDurbar()

def chat_function(message, history):
    """Chat function with streaming response"""
    if not message.strip():
        yield "Please enter a valid question about Pune's history."
        return
    
    try:
        response = get_ai_response(message)
        # Stream the response character by character
        for i in range(len(response)):
            time.sleep(0.005)
            yield response[: i + 1]
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        yield "I apologize, but I encountered an error. Please try again."

# Load avatar images with fallback
def get_avatar_images():
    user_img = "images/user.png" if os.path.exists("images/user.png") else None
    bot_img = "images/bot.png" if os.path.exists("images/bot.png") else None
    return (user_img, bot_img)

# Create the Gradio interface
def create_app():
    with gr.Blocks(
        title="Mastani.ai",
        theme=midnight_theme,
        analytics_enabled=False,
        js="() => { window.scrollTo(0, 0); }"
    ) as demo:
        gr.HTML(header_html)
        
        # Add a status message
        with gr.Row():
            gr.HTML("<p style='text-align: center; color: #a0a0a0;'>🔄 Initializing historical database... First query may take a moment.</p>")
        
        gr.ChatInterface(
            fn=chat_function,
            type="messages",
            chatbot=gr.Chatbot(
                height=600,
                show_label=False,
                avatar_images=get_avatar_images()
            ),
            examples=[
                "Who made Pune the capital during the 18th century?",
                "Which two rivers meet at the 'Sangam' in Pune?",
                "What is the name of Pune's oldest area where the city originally began?"
            ],
            cache_examples=False
        )
    
    return demo

# Initialize database creation on startup
def initialize_app():
    """Initialize the application"""
    logger.info("Initializing Mastani.ai application...")
    
    # Pre-create the vector database in the background
    import threading
    def create_db_async():
        logger.info("Starting database creation in background...")
        create_qa_chain()
        logger.info("Database initialization completed!")
    
    # Start database creation in background thread
    db_thread = threading.Thread(target=create_db_async)
    db_thread.daemon = True
    db_thread.start()

# Create the app
demo = create_app()

# For Render deployment
if __name__ == "__main__":
    initialize_app()
    logger.info(f"Starting app on {HOST}:{PORT}")
    demo.launch(
        server_name=HOST,
        server_port=PORT,
        share=False,
        show_error=True,
        quiet=False
    )
