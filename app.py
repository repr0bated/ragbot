"""
Entry point for the RAG Chatbot application
For Hugging Face Spaces deployment
"""

from gradio_app import demo

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch() 