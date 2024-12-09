from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import ChatRequest, ChatResponse
from .rag_service import RAGService

app = FastAPI()

# Configure CORS - Updated configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, you can use "*". For production, specify your domain
    allow_credentials=True,  # Changed to False since we're using "*" for origins
    allow_methods=["*"],  # Explicitly specify allowed methods
    allow_headers=["*"],  # Allow all headers
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Initialize RAG service
rag_service = RAGService()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = await rag_service.get_response(
        message=request.message,
        chat_history=request.chat_history
    )
    return ChatResponse(response=response)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
