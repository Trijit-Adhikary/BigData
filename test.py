pip install langchain langchain-openai langchain-community


# Request
{
    "query": "What is machine learning?",
    "max_tokens": 800,
    "temperature": 0.2
}

# Response
{
    "response": "Machine learning is a subset of artificial intelligence...",
    "followup_qs": [
        "What are the different types of machine learning algorithms?",
        "How do I choose the right machine learning model for my data?",
        "What are some real-world applications of machine learning?"
    ]
}
















from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import os
import json

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# LANGCHAIN-ONLY imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

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

# LangChain GLOBAL Clients
langchain_chat_client = None
langchain_embeddings_client = None
azure_search_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LangChain clients on startup and Cleanup on shutdown"""
    global langchain_chat_client, langchain_embeddings_client, azure_search_client
    
    try:
        # CORRECTED: Initialize LangChain Azure Chat OpenAI client
        langchain_chat_client = AzureChatOpenAI(
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2025-01-01-preview",
            azure_deployment="gpt-4o",  # ✅ Correct parameter name
            temperature=0.1
            # ❌ Don't put max_tokens here - it goes in invoke()
        )

        # CORRECTED: Initialize LangChain Azure Embeddings client
        langchain_embeddings_client = AzureOpenAIEmbeddings(
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2025-01-01-preview",
            azure_deployment="text-embedding-ada-002",  # ✅ Use azure_deployment
            model="text-embedding-ada-002"
        )

        # Initialize Azure Search client
        azure_search_client = SearchClient(
            endpoint="AZURE_SEARCH_ENDPOINT",
            index_name="rag-hellopdf",
            credential=AzureKeyCredential("AZURE_SEARCH_ADMIN_KEY")
        )
        
        print("✅ LangChain Azure clients initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        raise

    yield
    # Clean up resources
    if azure_search_client:
        await azure_search_client.close()
    print("🔄 Application shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title="Pure LangChain PDF RAG Chatbot API",
    description="FastAPI application using exclusively LangChain for Azure OpenAI",
    version="0.123.0",
    lifespan=lifespan
)

async def langchain_query_vectorizer(query: str, embeddings_client):
    """Convert query to vector embedding using LangChain"""
    try:
        # Check if async method exists
        if hasattr(embeddings_client, 'aembed_query'):
            embedding = await embeddings_client.aembed_query(query)
        else:
            # Use sync method (LangChain will handle it)
            embedding = embeddings_client.embed_query(query)
        return embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        # Fallback to sync method
        embedding = embeddings_client.embed_query(query)
        return embedding

async def langchain_vector_search(query: str, embeddings_client, search_client):
    """Perform vector search using LangChain embeddings"""
    try:
        query_vector = await langchain_query_vectorizer(query, embeddings_client)

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=5,
            fields="text_vector"
        )

        results = await search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name="rag-hellopdf-semantic-configuration",
            top=5,
            select=["chunk"]
        )

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")

async def generate_langchain_only_response(user_query: str, max_tokens: int, temperature: float):
    """Generate response using ONLY LangChain components"""
    try:
        # Get relevant chunks using LangChain embeddings
        chunks = await langchain_vector_search(user_query, langchain_embeddings_client, azure_search_client)
        
        # Convert search results to context string
        context_chunks = []
        async for chunk in chunks:
            context_chunks.append(chunk.get("chunk", ""))
        
        current_context = "\n".join(context_chunks)

        # CORRECTED: Create a new client instance with updated parameters
        dynamic_chat_client = AzureChatOpenAI(
            azure_endpoint=langchain_chat_client.azure_endpoint,
            api_key=langchain_chat_client.api_key,
            api_version=langchain_chat_client.api_version,
            azure_deployment=langchain_chat_client.azure_deployment,  # ✅ Correct parameter
            temperature=temperature
            # ❌ max_tokens goes in invoke(), not here
        )

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

        Return your response in the following JSON format:
        {{
            "response": "Your detailed answer here",
            "followup_qs": ["question 1", "question 2", "question 3"]
        }}

        {format_instructions}
        """

        # Create prompt template using LangChain
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

        # CORRECTED: Generate response with max_tokens in invoke()
        response = await dynamic_chat_client.ainvoke(
            formatted_prompt.to_messages(),
            config={"max_tokens": max_tokens}  # ✅ Pass max_tokens here
        )
        
        # Parse the structured response
        try:
            parsed_response = parser.parse(response.content)
            return {
                "response": parsed_response.response,
                "followup_qs": parsed_response.followup_qs
            }
        except OutputParserException:
            # Robust fallback parsing
            content = response.content.strip()
            
            # Try to extract JSON
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            try:
                parsed_json = json.loads(content)
                
                if "response" in parsed_json and "followup_qs" in parsed_json:
                    followup_qs = parsed_json["followup_qs"][:3]
                    while len(followup_qs) < 3:
                        followup_qs.append(f"What else would you like to know about {user_query}?")
                    
                    return {
                        "response": parsed_json["response"],
                        "followup_qs": followup_qs
                    }
            except json.JSONDecodeError:
                pass
            
            # Final fallback
            return {
                "response": response.content,
                "followup_qs": [
                    f"What are the key aspects of {user_query}?",
                    f"How does {user_query} relate to other concepts?",
                    f"Can you provide examples related to {user_query}?"
                ]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Pure LangChain Azure OpenAI RAG Chatbot API is running!",
        "status": "healthy",
        "using": "LangChain exclusively for Azure OpenAI"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    LangChain-only chat endpoint
    
    - **query**: User's question or message
    - **max_tokens**: Maximum tokens in response (default: 1000)  
    - **temperature**: Response creativity (0.0-1.0, default: 0.1)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await generate_langchain_only_response(
            request.query, 
            request.max_tokens, 
            request.temperature
        )
        
        return ChatResponse(
            response=result["response"],
            followup_qs=result["followup_qs"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat/raw")
async def chat_raw_endpoint(request: ChatRequest):
    """Returns exact JSON structure using LangChain only"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await generate_langchain_only_response(
            request.query, 
            request.max_tokens, 
            request.temperature
        )
        
        return {
            "response": result["response"],
            "followup_qs": result["followup_qs"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

