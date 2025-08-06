# --- PATCH for asyncio event loop in server threads ---
import nest_asyncio

# --- All necessary imports ---
import gradio as gr
import time
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from gradio.themes.base import Base

# --- CONFIGURATION ---
load_dotenv()
DB_PATH = "pune_vector_db"

# Load the header HTML from the templates folder
try:
    with open("templates/header.html", "r", encoding="utf-8") as file:
        header_html = file.read()
except FileNotFoundError:
    print("Warning: header.html not found. Using a default title.")
    header_html = "<h1 style='text-align: center;'>Mastani.ai</h1>"


# --- BACKEND LOGIC (Merged from backend.py) ---
def create_qa_chain():
    """Initializes and returns a RetrievalQA chain."""
    print("--- Initializing Google AI models and vector store... ---")
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
    print("--- QA Chain created successfully. ---")
    return chain

# --- Create a single, reusable QA chain instance when the app starts ---
# This is for performance and stability. The patch is applied here.
nest_asyncio.apply()
qa_chain = create_qa_chain()


def get_ai_response(question: str):
    """Gets a response from the reusable QA chain."""
    # This now uses the pre-loaded chain, making it much faster.
    result = qa_chain.invoke({"query": question})
    return result["result"]

# --- CUSTOM THEME ---
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

# --- CHAT FUNCTION (for Gradio) ---
def chat_function(message, history):
    """Yields the AI response for a streaming effect."""
    response = get_ai_response(message)
    for i in range(len(response)):
        time.sleep(0.005)
        yield response[: i + 1]

# --- GRADIO INTERFACE DEFINITION ---
with gr.Blocks(
    title="Mastani.ai",
    theme=midnight_theme,
    analytics_enabled=False,
    js="() => { window.scrollTo(0, 0); }"
) as demo:
    gr.HTML(header_html)
    
    gr.ChatInterface(
        fn=chat_function,
        type="messages",
        chatbot=gr.Chatbot(
            height=600,
            show_label=False,
            avatar_images=("images/user.png", "images/bot.png")
        ),
        examples=[
            "Who made Pune the capital during the 18th century?",
            "Which two rivers meet at the ‘Sangam’ in Pune?",
            "What is the name of Pune’s oldest area where the city originally began?"
        ],
        cache_examples=False
    )

# --- LAUNCH THE APP ---
if __name__ == "__main__":
    # This configuration is correct for deployment on Render.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv('PORT', 7860))
    )
