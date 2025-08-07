import gradio as gr
import time
import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gradio.themes.base import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PORT = int(os.environ.get("PORT", 7860))
DB_PATH = "pune_vector_db"
qa_chain = None
is_initializing = True

def load_header_html():
    try:
        with open("templates/header.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "<h1>🏛️ Mastani.ai - Historical Inquiry System</h1>"

header_html = load_header_html()

def load_documents():
    documents = []
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        return documents
    
    try:
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            if filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []

def create_vector_database():
    try:
        if os.path.exists(DB_PATH):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            return vector_store
        
        documents = load_documents()
        if not documents:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        return None

def create_qa_chain():
    global qa_chain, is_initializing
    
    if qa_chain is not None:
        return qa_chain
    
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
        
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
        is_initializing = False
        return qa_chain
        
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        is_initializing = False
        return None

def get_ai_response(question: str):
    global is_initializing
    
    try:
        if is_initializing:
            create_qa_chain()
        
        if qa_chain is None:
            return "Historical database is currently unavailable. Please check if your data files are present and try again."
        
        result = qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        return "I encountered an error while processing your question. Please try rephrasing your query."

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
    if not message.strip():
        yield "Please enter a valid question about Pune's history."
        return
    
    try:
        response = get_ai_response(message)
        for i in range(len(response)):
            time.sleep(0.003)
            yield response[: i + 1]
    except Exception as e:
        logger.error(f"Error in chat function: {e}")
        yield "I encountered an error. Please try again."

def get_avatar_images():
    user_img = "images/user.png" if os.path.exists("images/user.png") else None
    bot_img = "images/bot.png" if os.path.exists("images/bot.png") else None
    return (user_img, bot_img)

with gr.Blocks(
    title="Mastani.ai",
    theme=midnight_theme,
    analytics_enabled=False,
    js="() => { window.scrollTo(0, 0); }"
) as demo:
    gr.HTML(header_html)
    
    with gr.Row():
        gr.HTML("<p style='text-align: center; color: #a0a0a0;'>📚 Ask me anything about Pune's rich history!</p>")
    
    gr.ChatInterface(
        fn=chat_function,
        type="messages",
        chatbot=gr.Chatbot(
            height=600,
            show_label=False,
            avatar_images=get_avatar_images(),
            type="messages"
        ),
        examples=[
            "Who made Pune the capital during the 18th century?",
            "Which two rivers meet at the 'Sangam' in Pune?",
            "What is the name of Pune's oldest area where the city originally began?"
        ],
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True
    )
