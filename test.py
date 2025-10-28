from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import os
import json

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from openai import AsyncAzureOpenAI

# LangChain imports
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.output_parser import OutputParserException

# Pydantic models for request and response
class ChatRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.1

class ChatResponse(BaseModel):
    response: str
    followup_qs: List[str]

class LangChainResponse(BaseModel):
    response: str
    followup_qs: List[str]

# Azure GLOBAL Clients -> Initialize on startup
azure_model_client = None
azure_search_client = None
langchain_client = None
embd_model = "text-embedding-ada-002"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Azure clients on startup and Cleanup on shutdown"""
    global azure_model_client, azure_search_client, langchain_client
    
    try:
        # Initialize Azure OpenAI client
        azure_model_client = AsyncAzureOpenAI(
            api_key="AZURE_OPENAI_KEY",
            api_version="2025-01-01-preview",
            azure_endpoint="AZURE_OPENAI_ENDPOINT"
        )

        # Initialize LangChain Azure OpenAI client
        langchain_client = AzureChatOpenAI(
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2025-01-01-preview",
            deployment_name="gpt-4o",
            temperature=0.1
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
    title="LangChain PDF RAG Chatbot API",
    description="FastAPI application for PDF RAG Chatbot with LangChain integration",
    version="0.121.0",
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

    results = await azure_search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="rag-hellopdf-semantic-configuration",
        top=5,
        select=["chunk"]
    )

    return results

async def generate_langchain_response(user_query: str, max_tokens: int, temperature: float, azure_model_client, azure_search_client, embd_model):
    """Generate chat response with follow-up questions using LangChain"""
    try:
        # Get relevant chunks from vector search
        chunks = await vector_search(user_query, azure_model_client, azure_search_client, embd_model)
        
        # Convert search results to context string
        context_chunks = []
        async for chunk in chunks:
            context_chunks.append(chunk.get("chunk", ""))
        
        current_context = "\n".join(context_chunks)

        # Configure LangChain client with dynamic parameters
        langchain_client.temperature = temperature
        langchain_client.max_tokens = max_tokens

        # Create output parser for structured response
        parser = PydanticOutputParser(pydantic_object=LangChainResponse)

        # System prompt template
        system_template = """
        You are a helpful AI assistant who understands the user query carefully and provides comprehensive answers based on the current context.

        Current context:
        {context}

        Instructions:
        1. Answer the user's question based on the provided context
        2. Keep your answer informative yet concise
        3. Generate exactly 3 relevant follow-up questions that would help the user explore the topic deeper
        4. Follow-up questions should be specific and actionable based on the context

        {format_instructions}
        """

        # Create prompt template
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{query}")
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

        # Format the prompt
        formatted_prompt = chat_prompt.format_prompt(
            context=current_context,
            query=user_query,
            format_instructions=parser.get_format_instructions()
        )

        # Generate response using LangChain
        response = await langchain_client.ainvoke(formatted_prompt.to_messages())
        
        # Parse the structured response
        try:
            parsed_response = parser.parse(response.content)
            return {
                "response": parsed_response.response,
                "followup_qs": parsed_response.followup_qs
            }
        except OutputParserException:
            # Fallback: Try to extract JSON from response
            try:
                # Look for JSON in the response content
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:-3].strip()
                elif content.startswith('```'):
                    content = content[3:-3].strip()
                
                parsed_json = json.loads(content)
                
                # Validate required fields
                if "response" in parsed_json and "followup_qs" in parsed_json:
                    # Ensure followup_qs is a list with exactly 3 questions
                    followup_qs = parsed_json["followup_qs"][:3]  # Take first 3
                    while len(followup_qs) < 3:  # Pad if less than 3
                        followup_qs.append(f"Can you tell me more about {user_query}?")
                    
                    return {
                        "response": parsed_json["response"],
                        "followup_qs": followup_qs
                    }
                else:
                    raise ValueError("Invalid JSON structure")
                    
            except (json.JSONDecodeError, ValueError):
                # Final fallback: Create structured response manually
                return {
                    "response": response.content,
                    "followup_qs": [
                        f"What are the key details about {user_query}?",
                        f"How does {user_query} relate to other topics?",
                        f"Can you provide more examples related to {user_query}?"
                    ]
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "LangChain Vector Search Chatbot API is running!", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for LangChain-powered conversations with follow-up questions
    
    - **query**: User's question or message
    - **max_tokens**: Maximum tokens in response (default: 1000)
    - **temperature**: Response creativity (0.0-1.0, default: 0.1)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await generate_langchain_response(
            request.query, 
            request.max_tokens, 
            request.temperature,
            azure_model_client, 
            azure_search_client,
            embd_model
        )
        
        return ChatResponse(
            response=result["response"],
            followup_qs=result["followup_qs"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Additional endpoint for testing structured output
@app.post("/chat/raw")
async def chat_raw_endpoint(request: ChatRequest):
    """
    Raw endpoint that returns the exact JSON structure as specified
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await generate_langchain_response(
            request.query, 
            request.max_tokens, 
            request.temperature,
            azure_model_client, 
            azure_search_client,
            embd_model
        )
        
        # Return exact JSON structure as requested
        return {
            "response": result["response"],
            "followup_qs": result["followup_qs"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
