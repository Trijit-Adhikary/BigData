from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import os

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from openai import AsyncAzureOpenAI

# Pydentic models for request and response
class ChatRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.1

class ChatResponse(BaseModel):
    response: str
    context_chunks_count: int


# Azure GLOBAL Clients -> Initialize on startup
azure_model_client = None
azure_search_client = None
embd_model = "text-embedding-ada-002"



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Azure clients on startup and Cleanup on shutdown"""
    global azure_model_client, azure_search_client
    
    try:
        # Initialize Azure OpenAI client
        azure_model_client = AsyncAzureOpenAI(
            api_key="AZURE_OPENAI_KEY",
            api_version="2025-01-01-preview",
            azure_endpoint="AZURE_OPENAI_ENDPOINT"
        )

        # Initialize Azure Search client
        azure_search_client = SearchClient(
            endpoint="AZURE_SEARCH_ENDPOINT",
            index_name="rag-hellopdf",
            credential=AzureKeyCredential("AZURE_SEARCH_ADMIN_KEY")
        )
        
        print("✅ Azure clients initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        raise

    yield
    # Clean up the ML models and release the resources
    if azure_search_client:
        await azure_search_client.close()
    print("🔄 Application shutdown complete")




# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG Chatbot API",
    description="FastAPI application for PDF RAG Chatbot",
    version="0.120.0",
    lifespan=lifespan
)


async def query_vectorizer(query: str, azure_model_client, embd_model: str):
    """Convert query to vector embedding"""
    response = await azure_model_client.embeddings.create(
        input=query,
        model=embd_model
    )
    return response.data[0].embedding


async def vector_search(query: str, azure_model_client, azure_search_client, embd_model: str):
    """Perform vector search on Azure AI Search Index"""
    query_vector = await query_vectorizer(query, azure_model_client, embd_model)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=5,
        fields="text_vector"
    )

    # async with azure_search_client:
    results = await azure_search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="rag-hellopdf-semantic-configuration",
        top=5,
        select=["chunk"]
    )

    return results


async def generate_chat_response(user_query: str, max_tokens: int, temperature: float, azure_model_client, azure_search_client, embd_model):
    """Generate chat response using Azure OpenAI"""
    try:
        # Get relevant chunks from vector search
        chunks = await vector_search(user_query, azure_model_client, azure_search_client, embd_model)
        
        # Convert search results to context string
        context_chunks = []
        chunk_count = 0
        async for chunk in chunks:
            context_chunks.append(chunk.get("chunk", ""))
            chunk_count += 1
        
        current_context = "\n".join(context_chunks)

        sys_prompt = f"""
        You are a helpful AI assistant who understands the user query carefully and then answers the question based on the current context.

        current context:
        {current_context}

        Keep your answers short and precise
        Ask follow up questions to the user to help the user dig deeper
        """

        messages = []

        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_query})

        # Generate the completion asynchronously
        completion = await azure_model_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95
        )

        return completion.choices[0].message.content, chunk_count

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Vector Search Chatbot API is running!", "status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for vector search-powered conversations
    
    - **query**: User's question or message
    - **max_tokens**: Maximum tokens in response (default: 500)
    - **temperature**: Response creativity (0.0-1.0, default: 0.1)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response_text, chunk_count = await generate_chat_response(
            request.query, 
            request.max_tokens, 
            request.temperature,
            azure_model_client, 
            azure_search_client,
            embd_model
        )
        
        return ChatResponse(
            response=response_text,
            context_chunks_count=chunk_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
