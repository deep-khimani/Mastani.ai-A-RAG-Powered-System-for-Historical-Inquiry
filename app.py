import gradio as gr
import time
import os
from gradio.themes.base import Base
# --- KEY CHANGE: Import the AI function directly from your backend ---
from backend import get_ai_response

# --- CONFIGURATION ---
# Load the header HTML from the templates folder
with open("templates/header.html", "r", encoding="utf-8") as file:
    header_html = file.read()

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
    """Yields the AI response character by character for a streaming effect."""
    # This now calls the imported function from backend.py
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
