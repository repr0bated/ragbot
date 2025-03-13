# RAG Chatbot with Pinecone and Llama-Instruct

A Retrieval-Augmented Generation (RAG) chatbot system that uses Pinecone for vector storage and open-source Llama-Instruct models for generation. This version is optimized for Paperspace environments and includes a Gradio interface for Hugging Face Spaces deployment.

## Features

- Processes and embeds conversation data using sentence transformers
- Extracts rich metadata including dates, people, and sources from text
- Stores vector embeddings in Pinecone for fast similarity search
- Uses free open-source Llama-Instruct models for text generation
- Provides a responsive Gradio web interface
- Optimized for Paperspace environments
- Deployable on Hugging Face Spaces

## Prerequisites

- Python 3.8+
- Pinecone API key
- Paperspace storage (optional)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd ragbot
```

2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys (use `.env.example` as a template):
```
cp .env.example .env
# Edit .env with your actual API keys
```

## Usage

The application has several modes of operation:

### 1. Process and Store Data (CLI)

To process your conversations and store them in Pinecone:
```
python rag-chatbot.py
```
Then select option 2 and provide the path to your conversations JSON file.

### 2. Start the Web Interface (Gradio)

To start the chatbot with the Gradio web interface:
```
python app.py
```
The web interface will be available at http://localhost:7860.

### 3. Process Paperspace Data

If you're running in a Paperspace environment with mounted storage:
```
python rag-chatbot.py
```
Then select option 4 to process the data from the mounted storage.

## Paperspace Integration

This application is designed to work seamlessly with Paperspace environments. It automatically detects if it's running on Paperspace and uses the mounted storage at `/storage/datasets/`.

## Model Information

The application uses the following models:

- **Embedding Model**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Text Generation**: Llama-Instruct model from Hugging Face (default: `NousResearch/Llama-2-7b-chat-hf`)

You can change the model by setting the `MODEL_NAME` environment variable in the `.env` file.

## Hugging Face Spaces Deployment

To deploy to Hugging Face Spaces:

1. Create a new Space on Hugging Face with SDK: Gradio
2. Connect your repository to the Space
3. Add your Pinecone API key as a secret in the Space settings

## Data Format

The application expects conversations in a JSON format with the following structure:
```json
[
  {
    "id": "conversation1",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      },
      {
        "role": "assistant",
        "content": "I'm doing well, thank you!"
      }
    ]
  }
]
```

## License

[MIT License](LICENSE)