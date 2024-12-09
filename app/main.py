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
    allow_origins=["https://techbites.xyz","web-production-30a7.up.railway.app"],  # Update this with your portfolio domain in production
    allow_credentials=False,  # Set to False when using "*" for origins
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
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
