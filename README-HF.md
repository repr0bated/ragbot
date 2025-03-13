# RAG Chatbot with Llama-Instruct

A Retrieval-Augmented Generation (RAG) chatbot that uses Pinecone for vector storage and the open-source Llama-Instruct model for text generation.

## About

This chatbot uses:

- **Sentence Transformers** for embedding generation
- **Pinecone** for vector similarity search
- **Llama-Instruct** for generating natural language responses
- **Gradio** for the UI

## Setup

This space requires a Pinecone API key to be configured as a secret. Make sure to:

1. Set `PINECONE_API_KEY` in the Space secrets
2. Set `PINECONE_ENVIRONMENT` in the Space secrets (e.g., "gcp-starter")

## Usage

Simply type your query in the chat interface and the system will:

1. Convert your query to a vector embedding
2. Find the most relevant information from the Pinecone database
3. Format a prompt with the retrieved information
4. Generate a response using Llama-Instruct
5. Show you the sources of information used in the response

## Model Information

By default, this space uses:
- `all-MiniLM-L6-v2` for embeddings
- `NousResearch/Llama-2-7b-chat-hf` for text generation

You can customize the model by setting the `MODEL_NAME` environment variable. 