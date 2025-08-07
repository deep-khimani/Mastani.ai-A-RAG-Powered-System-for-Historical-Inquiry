import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()
PORT = int(os.environ.get("PORT", 7860))

def chat_function(message, history):
    return f"Echo: {message} (Database initialization in progress...)"

demo = gr.ChatInterface(
    fn=chat_function,
    title="Mastani.ai - Historical Inquiry System",
    description="Ask questions about Pune's history"
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False
    )
