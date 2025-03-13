"""
Gradio UI for the RAG Chatbot
Provides a web interface for the chatbot, deployable on Hugging Face Spaces
"""

import os
import gradio as gr
import json
import logging
from dotenv import load_dotenv
from llama_model import LlamaInstructModel
from sentence_transformers import SentenceTransformer
import re
import pinecone
from typing import List, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ragbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ragbot_gradio")

# Load environment variables
load_dotenv()

# Constants and configurations
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = "gpt-conversations"
MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"

class DataProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text
    
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings for a text"""
        return self.model.encode(text)

class PineconeClient:
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self._initialize()
        
    def _initialize(self):
        try:
            logger.info(f"Initializing Pinecone: {self.environment}")
            pinecone.init(api_key=self.api_key, environment=self.environment)
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            raise
    
    def query(self, vector: List[float], top_k: int = 3) -> Dict:
        """Query the index with a vector"""
        try:
            return self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            return {"matches": []}

class RAGChatbot:
    def __init__(self):
        logger.info("Initializing RAG Chatbot")
        self.processor = DataProcessor()
        self.pinecone_client = PineconeClient(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT,
            index_name=PINECONE_INDEX
        )
        self.llm = None
    
    def _get_llm(self):
        """Lazy initialization of the language model"""
        if self.llm is None:
            logger.info(f"Loading LLM: {MODEL_NAME}")
            self.llm = LlamaInstructModel(model_name=MODEL_NAME)
        return self.llm
    
    def process_query(self, query: str, history: List = None):
        """Process a query using RAG and return a response"""
        if history is None:
            history = []
            
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Generate embedding for the query
            processed_query = self.processor.preprocess_text(query)
            query_embedding = self.processor.generate_embeddings(processed_query).tolist()
            
            # Retrieve relevant documents
            results = self.pinecone_client.query(vector=query_embedding, top_k=3)
            
            # Extract contexts
            contexts = []
            for match in results.get("matches", []):
                if "metadata" in match and "text" in match["metadata"]:
                    contexts.append(match["metadata"]["text"])
            
            # Format system and user prompts
            system_prompt = "You are a helpful assistant. Answer the question accurately based on the provided context."
            
            user_prompt = f"""
            Answer based on the following contexts:
            
            {' '.join(contexts)}
            
            Question: {query}
            """
            
            # Generate response
            llm = self._get_llm()
            response = llm.generate_response(system_prompt, user_prompt)
            
            # Format sources as markdown for display
            sources_md = "\n\n**Sources:**\n"
            for i, match in enumerate(results.get("matches", [])[:3]):
                if "metadata" in match:
                    score = float(match.get("score", 0)) * 100
                    text = match["metadata"].get("text", "")
                    # Truncate long texts
                    if len(text) > 200:
                        text = text[:200] + "..."
                    sources_md += f"- **Source {i+1}** (Relevance: {score:.1f}%): {text}\n"
            
            # Append sources to response
            if results.get("matches"):
                response += sources_md
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error while processing your query. Please try again."

# Initialize the chatbot
chatbot = RAGChatbot()

# Define the Gradio chat interface
def respond(message, history):
    """Handle incoming messages and generate responses"""
    return chatbot.process_query(message, history)

# Create the Gradio interface
demo = gr.ChatInterface(
    respond,
    title="RAG Chatbot",
    description="A Retrieval-Augmented Generation chatbot using Llama Instruct and Pinecone",
    theme="soft",
    examples=[
        "What information do you have about machine learning?",
        "Tell me about natural language processing",
        "Explain the concept of retrieval-augmented generation"
    ],
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

# For Hugging Face Spaces deployment
if __name__ == "__main__":
    demo.launch(share=True) 