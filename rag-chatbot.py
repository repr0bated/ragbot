# RAG Chatbot System with Pinecone and Mobile Interface

# Part 1: Data Processing and Embedding

import json
import os
import re
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import datetime
from dateutil.parser import parse as date_parse
import email.utils

# Load environment variables
load_dotenv()

# Paperspace data configuration
PAPERSPACE_STORAGE = {
    "project_id": "t9hrtpxj9",
    "storage_id": "dsm97c6ujy7vf8",
    "storage_type": "s3",
    "storage_bucket": "t9hrtpxj9",
    "storage_access_key": "0DOF5K7A04RUSD5R1SC0"
}

# Configure data paths
# For Paperspace, data is typically mounted at /storage
PAPERSPACE_DATA_PATH = "/storage/datasets/combined_documents_deduplicated.json"
# Fallback to local data if not running on Paperspace
LOCAL_DATA_PATH = "combined_documents_deduplicated.json"

# Determine which data path to use
def get_data_path():
    if os.path.exists("/storage"):
        print("Using Paperspace mounted storage")
        return PAPERSPACE_DATA_PATH
    else:
        print("Using local storage")
        return LOCAL_DATA_PATH

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

class DataProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.stop_words = set(stopwords.words('english'))
    
    def load_conversations(self, file_path: str) -> List[Dict[str, Any]]:
        """Load conversations from a JSON file"""
        print(f"Loading data from: {file_path}")
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def extract_keywords(self, texts: List[str], top_n: int = 10) -> Dict[str, List[str]]:
        """Extract keywords using TF-IDF"""
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        feature_names = vectorizer.get_feature_names_out()
        keywords_dict = {}
        
        for i, doc in enumerate(texts):
            tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
            keywords_dict[i] = [word for word, score in sorted_scores[:top_n]]
            
        return keywords_dict
    
    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text using multiple patterns and formats.
        Returns a list of ISO format dates (YYYY-MM-DD).
        """
        if not text:
            return []
            
        # Common date patterns
        date_patterns = [
            # ISO format: YYYY-MM-DD
            r'\d{4}-\d{1,2}-\d{1,2}',
            # MM/DD/YYYY or DD/MM/YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',
            # Month name formats: Jan 1, 2023 or January 1st, 2023
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}',
            # DD-MM-YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',
            # Email date format: Tue, 01 Mar 2022 12:34:56
            r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',
        ]
        
        extracted_dates = []
        
        # Extract dates using patterns
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                try:
                    # Try to parse the date string
                    if ',' in date_str and any(day in date_str for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
                        # Handle email date format
                        parsed_date = email.utils.parsedate_to_datetime(date_str)
                        date_iso = parsed_date.strftime('%Y-%m-%d')
                    else:
                        # Handle other formats
                        parsed_date = date_parse(date_str, fuzzy=True)
                        date_iso = parsed_date.strftime('%Y-%m-%d')
                    
                    extracted_dates.append(date_iso)
                except (ValueError, OverflowError):
                    # Skip dates that can't be parsed
                    continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dates = [date for date in extracted_dates if not (date in seen or seen.add(date))]
        
        return unique_dates
    
    def extract_people(self, text: str) -> List[str]:
        """
        Extract potential people names from text.
        Uses a combination of regex patterns and NLTK Named Entity Recognition.
        """
        if not text:
            return []
            
        people = set()
        
        # Method 1: Use NLTK's named entity recognition
        try:
            sentences = nltk.sent_tokenize(text)
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(words)
                named_entities = nltk.ne_chunk(tagged)
                
                # Extract person names
                for chunk in named_entities:
                    if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                        name = ' '.join([c[0] for c in chunk])
                        if len(name.split()) >= 2:  # Only add if at least first and last name
                            people.add(name)
        except Exception:
            pass  # If NER fails, continue with regex patterns
                            
        # Method 2: Common name patterns (simplified)
        name_patterns = [
            # Title + Name: Mr. John Smith
            r'(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Sir|Madam|Lady)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
            # Name followed by credentials: John Smith, MD
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+,\s+(?:MD|PhD|JD|Esq)',
            # Names in quotes: "John Smith"
            r'"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+"',
            # Simple two-word capitalized names: John Smith
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        ]
        
        for pattern in name_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(0)
                # Clean up the name (remove titles, credentials, quotes)
                name = re.sub(r'^(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Sir|Madam|Lady)\.?\s+', '', name)
                name = re.sub(r',\s+(?:MD|PhD|JD|Esq)$', '', name)
                name = name.replace('"', '')
                people.add(name.strip())
        
        return list(people)
    
    def extract_sources(self, text: str) -> List[Dict[str, str]]:
        """
        Extract source information such as URLs, email addresses,
        references, and citations from text.
        """
        if not text:
            return []
            
        sources = []
        
        # URL pattern
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!./?=&+]*)?'
        urls = re.findall(url_pattern, text)
        for url in urls:
            sources.append({
                'type': 'url',
                'value': url
            })
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        for email in emails:
            sources.append({
                'type': 'email',
                'value': email
            })
        
        # Citation patterns
        citation_patterns = [
            # APA-like: (Author, 2023)
            r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}\)',
            # MLA-like: (Author 2023)
            r'\([A-Z][a-z]+\s+\d{4}\)',
            # IEEE-like: [1], [2], etc.
            r'\[\d+\]',
            # Footnote-like: [a], [b], etc. or superscript numbers
            r'\[[a-z]\]|\[\d+\]'
        ]
        
        for pattern in citation_patterns:
            citations = re.findall(pattern, text)
            for citation in citations:
                sources.append({
                    'type': 'citation',
                    'value': citation.strip()
                })
        
        return sources
    
    def extract_comprehensive_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from text including dates, 
        people, sources, and other useful information.
        """
        metadata = {}
        
        # Extract dates
        dates = self.extract_dates(text)
        if dates:
            metadata['dates'] = dates
            # Set the primary date (most recent)
            try:
                metadata['primary_date'] = sorted(dates)[-1]
            except:
                pass
        
        # Extract people
        people = self.extract_people(text)
        if people:
            metadata['people'] = people
            # The first person might be the author in many contexts
            metadata['potential_author'] = people[0]
        
        # Extract sources
        sources = self.extract_sources(text)
        if sources:
            metadata['sources'] = sources
        
        # Add text statistics
        metadata['word_count'] = len(text.split())
        metadata['char_count'] = len(text)
        
        # Add language detection (simple approach, just assuming English)
        metadata['language'] = 'english'
        
        # Add extraction timestamp
        metadata['metadata_extracted'] = datetime.datetime.now().isoformat()
        
        return metadata
    
    def categorize_conversations(self, texts: List[str], num_categories: int = 5) -> List[int]:
        """Categorize conversations using K-means clustering"""
        embeddings = self.model.encode(texts)
        kmeans = KMeans(n_clusters=num_categories, random_state=42)
        return kmeans.fit_predict(embeddings)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        return self.model.encode(texts)

class PineconeManager:
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = 384):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.initialize_pinecone()
        
    def initialize_pinecone(self):
        """Initialize Pinecone connection and index"""
        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        # Check if index exists, if not create it
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine"
            )
        
        self.index = pinecone.Index(self.index_name)
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """Insert or update vectors in the index"""
        self.index.upsert(vectors=vectors)
    
    def query(self, query_vector: List[float], top_k: int = 5) -> Dict:
        """Query the index with a vector"""
        return self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)

def count_conversations(json_file_path: str):
    """Count the number of conversations and messages in the JSON file"""
    # Initialize processor
    processor = DataProcessor()
    
    # Load conversations
    conversations = processor.load_conversations(json_file_path)
    
    # Count conversations
    num_conversations = len(conversations)
    
    # Count messages
    total_messages = 0
    for convo in conversations:
        if "messages" in convo:
            total_messages += len(convo["messages"])
    
    return {
        "num_conversations": num_conversations,
        "total_messages": total_messages
    }

def process_and_store_conversations(json_file_path: str = None):
    """Process conversations and store in Pinecone"""
    # Use provided path or get the default path
    if json_file_path is None:
        json_file_path = get_data_path()
    
    print(f"Processing data from: {json_file_path}")
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load conversations
    conversations = processor.load_conversations(json_file_path)
    
    # Count and print statistics
    stats = count_conversations(json_file_path)
    
    print(f"Processing {stats['num_conversations']} conversations with {stats['total_messages']} total messages")
    
    # Extract texts for processing
    texts = []
    for convo in conversations:
        if "messages" in convo:
            for message in convo["messages"]:
                if "content" in message:
                    texts.append(message["content"])
    
    # Preprocess texts
    print("Preprocessing texts...")
    preprocessed_texts = [processor.preprocess_text(text) for text in texts]
    
    # Extract keywords
    print("Extracting keywords...")
    keywords_dict = processor.extract_keywords(preprocessed_texts)
    
    # Categorize conversations
    print("Categorizing texts...")
    categories = processor.categorize_conversations(preprocessed_texts)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = processor.generate_embeddings(preprocessed_texts)
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    pinecone_manager = PineconeManager(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
        index_name="gpt-conversations"
    )
    
    # Prepare vectors for Pinecone
    print("Preparing vectors...")
    vectors = []
    for i, embedding in enumerate(embeddings):
        # Extract comprehensive metadata
        print(f"Extracting metadata for document {i+1}/{len(embeddings)}...", end="\r")
        enriched_metadata = processor.extract_comprehensive_metadata(texts[i])
        
        # Combine with existing metadata
        metadata = {
            "text": texts[i],
            "keywords": keywords_dict.get(i, []),
            "category": int(categories[i])
        }
        
        # Merge with enriched metadata
        metadata.update(enriched_metadata)
        
        vector = {
            "id": str(i),
            "values": embedding.tolist(),
            "metadata": metadata
        }
        vectors.append(vector)
    
    print("\nUploading vectors to Pinecone...")
    # Store vectors in batches (Pinecone has a limit)
    batch_size = 100
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    for i in range(0, len(vectors), batch_size):
        batch_num = i // batch_size + 1
        print(f"Uploading batch {batch_num}/{total_batches}...", end="\r")
        pinecone_manager.upsert_vectors(vectors[i:i+batch_size])
    
    print("\nData processing complete!")
    return "Data processed and stored successfully with enhanced metadata!"


# Part 2: RAG Chatbot Backend

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import openai
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates for the web interface
templates = Jinja2Templates(directory="templates")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize processor and Pinecone manager
processor = DataProcessor()
pinecone_manager = PineconeManager(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
    index_name="gpt-conversations"
)

# Define request and response models
class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]

@app.get("/", response_class=HTMLResponse)
async def get_web_interface(request: Request):
    """Serve the web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request using RAG"""
    try:
        # Preprocess query
        processed_query = processor.preprocess_text(request.query)
        
        # Generate embedding for the query
        query_embedding = processor.model.encode(processed_query).tolist()
        
        # Retrieve relevant documents from Pinecone
        results = pinecone_manager.query(query_vector=query_embedding, top_k=3)
        
        # Extract contexts from the results
        contexts = []
        for match in results.get("matches", []):
            if "metadata" in match and "text" in match["metadata"]:
                contexts.append(match["metadata"]["text"])
        
        # Construct prompt with retrieved contexts
        prompt = f"""
        Answer the following question based on the provided contexts. If the contexts don't contain relevant information, 
        say that you don't know but provide a general response if possible.
        
        Contexts:
        {' '.join(contexts)}
        
        Chat History:
        {' '.join([f"{msg['role']}: {msg['content']}" for msg in request.history])}
        
        Question: {request.query}
        """
        
        # Generate response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        # Extract and return the response
        answer = response.choices[0].message.content
        
        # Format sources for the response
        sources = []
        for i, match in enumerate(results.get("matches", [])[:3]):
            if "metadata" in match:
                sources.append({
                    "text": match["metadata"].get("text", ""),
                    "score": float(match.get("score", 0)),
                    "keywords": match["metadata"].get("keywords", []),
                    "category": match["metadata"].get("category", -1)
                })
        
        return ChatResponse(response=answer, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to start the server
def start_server():
    """Start the FastAPI server"""
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# Part 3: Frontend Web Interface (HTML, CSS, JS)

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Create index.html
index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Chatbot</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>RAG Chatbot</h1>
        </div>
        <div class="chat-messages" id="chat-messages">
            <div class="message bot">
                <div class="message-content">Hello! How can I help you today?</div>
            </div>
        </div>
        <div class="chat-input">
            <input type="text" id="user-input" placeholder="Type your message here...">
            <button id="send-button">Send</button>
        </div>
        <div class="sources-container" id="sources-container">
            <h3>Sources</h3>
            <div class="sources-list" id="sources-list"></div>
        </div>
    </div>
    <script src="/static/script.js"></script>
</body>
</html>"""

# Create styles.css
styles_css = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

body {
    background-color: #f5f5f5;
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.chat-container {
    width: 100%;
    max-width: 600px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}

.chat-header {
    padding: 15px;
    background-color: #4a6fa5;
    color: white;
    text-align: center;
}

.chat-messages {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
}

.message {
    margin-bottom: 15px;
    display: flex;
}

.user {
    justify-content: flex-end;
}

.bot {
    justify-content: flex-start;
}

.message-content {
    padding: 10px 15px;
    border-radius: 20px;
    max-width: 70%;
    word-break: break-word;
}

.user .message-content {
    background-color: #4a6fa5;
    color: white;
}

.bot .message-content {
    background-color: #e5e5e5;
    color: #333;
}

.chat-input {
    display: flex;
    padding: 15px;
    border-top: 1px solid #e5e5e5;
}

#user-input {
    flex: 1;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 20px;
    margin-right: 10px;
    outline: none;
}

#send-button {
    padding: 10px 20px;
    background-color: #4a6fa5;
    color: white;
    border: none;
    border-radius: 20px;
    cursor: pointer;
}

.sources-container {
    padding: 15px;
    border-top: 1px solid #e5e5e5;
    max-height: 200px;
    overflow-y: auto;
    display: none;
}

.sources-container h3 {
    margin-bottom: 10px;
    color: #4a6fa5;
}

.source-item {
    background-color: #f9f9f9;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
    border-left: 3px solid #4a6fa5;
}

.source-text {
    margin-bottom: 5px;
}

.source-meta {
    font-size: 0.8em;
    color: #666;
}

.keywords {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 5px;
}

.keyword {
    background-color: #e5e5e5;
    padding: 2px 5px;
    border-radius: 3px;
    font-size: 0.7em;
}

@media (max-width: 600px) {
    .chat-container {
        height: 100vh;
        border-radius: 0;
    }
}"""

# Create script.js
script_js = """document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const sourcesList = document.getElementById('sources-list');
    const sourcesContainer = document.getElementById('sources-container');
    
    let chatHistory = [];
    
    // Send message when Send button is clicked
    sendButton.addEventListener('click', sendMessage);
    
    // Send message when Enter key is pressed
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessage('user', message);
        
        // Clear input
        userInput.value = '';
        
        // Add message to history
        chatHistory.push({
            role: 'user',
            content: message
        });
        
        // Show loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot';
        loadingDiv.innerHTML = '<div class="message-content">Thinking...</div>';
        chatMessages.appendChild(loadingDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Send request to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message,
                history: chatHistory
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);
            
            // Add bot message to chat
            addMessage('bot', data.response);
            
            // Add message to history
            chatHistory.push({
                role: 'assistant',
                content: data.response
            });
            
            // Display sources if available
            if (data.sources && data.sources.length > 0) {
                displaySources(data.sources);
                sourcesContainer.style.display = 'block';
            } else {
                sourcesContainer.style.display = 'none';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // Remove loading indicator
            chatMessages.removeChild(loadingDiv);
            // Add error message
            addMessage('bot', 'Sorry, something went wrong. Please try again.');
        });
    }
    
    function addMessage(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function displaySources(sources) {
        // Clear previous sources
        sourcesList.innerHTML = '';
        
        // Add each source
        sources.forEach(source => {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'source-item';
            
            // Format text (truncate if too long)
            const text = source.text.length > 100 ? 
                source.text.substring(0, 100) + '...' : 
                source.text;
            
            // Create keywords HTML
            let keywordsHtml = '';
            if (source.keywords && source.keywords.length > 0) {
                keywordsHtml = '<div class="keywords">' + 
                    source.keywords.map(kw => `<span class="keyword">${kw}</span>`).join('') +
                    '</div>';
            }
            
            sourceDiv.innerHTML = `
                <div class="source-text">${text}</div>
                <div class="source-meta">
                    Relevance: ${(source.score * 100).toFixed(1)}% | 
                    Category: ${source.category}
                </div>
                ${keywordsHtml}
            `;
            
            sourcesList.appendChild(sourceDiv);
        });
    }
});"""

# Write files to disk
with open("templates/index.html", "w") as f:
    f.write(index_html)

with open("static/styles.css", "w") as f:
    f.write(styles_css)

with open("static/script.js", "w") as f:
    f.write(script_js)

# Part 4: Main Application

def main():
    """Main function to run the application"""
    print("RAG Chatbot System")
    print("1. Count records in conversations file")
    print("2. Process and store conversations")
    print("3. Start the web server")
    print("4. Process Paperspace mounted data")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        json_file = input("Enter the path to your conversations.json file: ")
        try:
            stats = count_conversations(json_file)
            print(f"\nStatistics for {json_file}:")
            print(f"Number of conversations: {stats['num_conversations']}")
            print(f"Total number of messages: {stats['total_messages']}")
            print(f"Average messages per conversation: {stats['total_messages']/stats['num_conversations']:.2f}")
        except Exception as e:
            print(f"Error counting records: {str(e)}")
    elif choice == "2":
        json_file = input("Enter the path to your conversations.json file: ")
        result = process_and_store_conversations(json_file)
        print(result)
    elif choice == "3":
        print("Starting web server...")
        start_server()
    elif choice == "4":
        print("Processing data from Paperspace mounted storage...")
        data_path = get_data_path()
        result = process_and_store_conversations(data_path)
        print(result)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
