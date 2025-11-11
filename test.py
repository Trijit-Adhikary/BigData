from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Annotated
import json
import uuid
from datetime import datetime, timedelta
import asyncio
import os
from dotenv import load_dotenv

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# LangChain 1.0.x imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableLambda,
    Runnable,
    RunnableConfig
)
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks import BaseCallbackHandler

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Pydantic models (v2.x compatible)
class ChatRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=1000, description="User's chat query")]
    session_id: Annotated[Optional[str], Field(None, description="Optional session identifier")]
    max_tokens: Annotated[Optional[int], Field(800, ge=100, le=4000, description="Maximum tokens for response")]
    temperature: Annotated[Optional[float], Field(0.1, ge=0.0, le=2.0, description="Temperature for response generation")]
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "session_id": "optional-session-id",
                "max_tokens": 800,
                "temperature": 0.1
            }
        }

class ChatResponse(BaseModel):
    response: Annotated[str, Field(description="AI assistant response")]
    followup_qs: Annotated[List[str], Field(description="Follow-up questions")]
    session_id: Annotated[str, Field(description="Session identifier")]
    metadata: Annotated[Dict[str, Any], Field(default_factory=dict, description="Additional metadata")]

class StructuredResponse(BaseModel):
    """Structured response model for LLM output parsing"""
    response: Annotated[str, Field(description="The main response to the user's query")]
    followup_qs: Annotated[List[str], Field(min_length=3, max_length=3, description="Exactly 3 follow-up questions")]

class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    total_messages: int
    created_at: str
    last_updated: str
    memory_window_size: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    last_updated: datetime
    message_count: int
    memory_window_size: int

# Modern in-memory session manager compatible with LangChain 1.0.x
class ModernMemoryManager:
    """Session manager using ConversationBufferWindowMemory for LangChain 1.0.x"""
    
    def __init__(self, memory_window_size: int = 5, session_timeout_hours: int = 24):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.memory_window_size = memory_window_size
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self._lock = asyncio.Lock()
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create new session with ConversationBufferWindowMemory"""
        session_id = session_id or str(uuid.uuid4())
        
        # Create ChatMessageHistory for the session
        chat_history = ChatMessageHistory()
        
        # Create ConversationBufferWindowMemory with LangChain 1.0.x configuration
        memory = ConversationBufferWindowMemory(
            k=self.memory_window_size,
            chat_memory=chat_history,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True,
            human_prefix="Human",
            ai_prefix="AI"
        )
        
        self.sessions[session_id] = {
            "memory": memory,
            "chat_history": chat_history,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "message_count": 0
        }
        
        return session_id
    
    async def get_session_memory(self, session_id: str) -> Optional[ConversationBufferWindowMemory]:
        """Get memory for session with timeout check"""
        async with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check if session expired
            if datetime.now() - session["last_accessed"] > self.session_timeout:
                await self._cleanup_session(session_id)
                return None
            
            # Update last accessed
            session["last_accessed"] = datetime.now()
            return session["memory"]
    
    async def get_session_chat_history(self, session_id: str) -> Optional[ChatMessageHistory]:
        """Get chat history for session"""
        async with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check if session expired
            if datetime.now() - session["last_accessed"] > self.session_timeout:
                await self._cleanup_session(session_id)
                return None
            
            session["last_accessed"] = datetime.now()
            return session["chat_history"]
    
    async def update_session(self, session_id: str):
        """Update session metadata after interaction"""
        async with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]["last_accessed"] = datetime.now()
                self.sessions[session_id]["message_count"] += 1
    
    async def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        async with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            return SessionInfo(
                session_id=session_id,
                created_at=session["created_at"],
                last_updated=session["last_accessed"],
                message_count=session["message_count"],
                memory_window_size=self.memory_window_size
            )
    
    async def _cleanup_session(self, session_id: str):
        """Remove expired session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    async def cleanup_expired_sessions(self) -> int:
        """Cleanup expired sessions"""
        async with self._lock:
            expired_sessions = []
            current_time = datetime.now()
            
            for session_id, session in self.sessions.items():
                if current_time - session["last_accessed"] > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            return len(expired_sessions)
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear session memory"""
        async with self._lock:
            if session_id in self.sessions:
                # Clear the memory
                self.sessions[session_id]["memory"].clear()
                self.sessions[session_id]["chat_history"].clear()
                self.sessions[session_id]["message_count"] = 0
                self.sessions[session_id]["last_accessed"] = datetime.now()
                return True
            return False
    
    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        await self.cleanup_expired_sessions()
        return len(self.sessions)
    
    async def list_active_sessions(self) -> List[SessionInfo]:
        """List all active sessions"""
        await self.cleanup_expired_sessions()
        async with self._lock:
            session_infos = []
            for session_id, session in self.sessions.items():
                session_infos.append(SessionInfo(
                    session_id=session_id,
                    created_at=session["created_at"],
                    last_updated=session["last_accessed"],
                    message_count=session["message_count"],
                    memory_window_size=self.memory_window_size
                ))
            return session_infos

# Callback handler for LangChain 1.0.x
class ConversationCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for conversation logging (LangChain 1.0.x compatible)"""
    
    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id
        self.start_time = None
    
    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any], 
        **kwargs: Any
    ) -> None:
        self.start_time = datetime.now()
        print(f"🔄 Chain started for session {self.session_id}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"✅ Chain completed for session {self.session_id} in {duration.total_seconds():.2f}s")
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        print(f"❌ Chain error for session {self.session_id}: {error}")

# Modern LangChain Chain compatible with 1.0.x
class ModernMemoryRAGChain:
    """LangChain 1.0.x compatible chain using ConversationBufferWindowMemory"""
    
    def __init__(self, llm: AzureChatOpenAI, session_id: str, memory_manager: ModernMemoryManager):
        self.llm = llm
        self.session_id = session_id
        self.memory_manager = memory_manager
        
        # Prompt template compatible with LangChain 1.0.x
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent AI assistant with conversation memory. 
Your responses should be contextually aware, referencing previous conversation when relevant.

Current context from knowledge base:
{context}

Instructions:
- Provide accurate, helpful responses based on context and conversation history
- Reference previous interactions when relevant
- Be concise but comprehensive
- Generate exactly 3 relevant follow-up questions

Respond in valid JSON format:
{format_instructions}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Output parser for LangChain 1.0.x
        self.output_parser = PydanticOutputParser(pydantic_object=StructuredResponse)
    
    async def ainvoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Async invocation compatible with LangChain 1.0.x"""
        try:
            # Get memory for the session
            memory = await self.memory_manager.get_session_memory(self.session_id)
            if memory is None:
                raise HTTPException(status_code=404, detail="Session not found or expired")
            
            # Get chat history from memory
            memory_variables = memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            # Create the runnable chain for LangChain 1.0.x
            def add_format_instructions(inputs: Dict[str, Any]) -> Dict[str, Any]:
                inputs["format_instructions"] = self.output_parser.get_format_instructions()
                return inputs
            
            # Build chain using LangChain 1.0.x patterns
            chain = (
                RunnableLambda(add_format_instructions)
                | self.prompt 
                | self.llm 
                | self.output_parser
            )
            
            # Prepare input with chat history
            chain_input = {
                "input": input_data["input"],
                "context": input_data.get("context", ""),
                "chat_history": chat_history
            }
            
            # Add callback handler if config provided
            if config is None:
                config = RunnableConfig()
            
            callback_handler = ConversationCallbackHandler(self.session_id)
            if config.get("callbacks"):
                config["callbacks"].append(callback_handler)
            else:
                config["callbacks"] = [callback_handler]
            
            # Invoke chain
            result = await chain.ainvoke(chain_input, config)
            
            # Save the interaction to memory
            memory.save_context(
                {"input": input_data["input"]},
                {"output": result.response}
            )
            
            # Update session
            await self.memory_manager.update_session(self.session_id)
            
            return {
                "response": result.response,
                "followup_qs": result.followup_qs
            }
            
        except Exception as e:
            print(f"❌ Chain invocation error: {e}")
            # Fallback response
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "followup_qs": [
                    "Could you please rephrase your question?",
                    "Is there something specific you'd like to know?",
                    "How else can I assist you today?"
                ]
            }

# Global clients and memory manager
langchain_chat_client: Optional[AzureChatOpenAI] = None
langchain_embeddings_client: Optional[AzureOpenAIEmbeddings] = None
azure_search_client: Optional[SearchClient] = None
memory_manager: Optional[ModernMemoryManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan management for LangChain 1.0.x"""
    global langchain_chat_client, langchain_embeddings_client, azure_search_client, memory_manager
    
    try:
        # Load configuration
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        
        if not all([azure_openai_endpoint, azure_openai_key]):
            raise ValueError("Missing required Azure OpenAI environment variables")
        
        # Initialize memory manager
        memory_manager = ModernMemoryManager(
            memory_window_size=5,  # Remember last 5 exchanges
            session_timeout_hours=24
        )
        
        # Initialize LangChain 1.0.x clients
        langchain_chat_client = AzureChatOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version="2024-06-01",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            temperature=0.1,
            max_tokens=None,  # LangChain 1.0.x pattern
            model_kwargs={
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        )

        langchain_embeddings_client = AzureOpenAIEmbeddings(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version="2024-06-01",
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            chunk_size=1000
        )

        azure_search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX", "rag-hellopdf"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        )
        
        print("✅ LangChain 1.0.x application initialized successfully")
        
        # Start background tasks
        cleanup_task = asyncio.create_task(periodic_maintenance())
        
        yield
        
        # Graceful shutdown
        print("🔄 Starting graceful shutdown...")
        cleanup_task.cancel()
        
        # Close resources
        if azure_search_client:
            await azure_search_client.close()
        
        print("✅ Application shutdown completed")
        
    except Exception as e:
        print(f"❌ Application initialization error: {e}")
        raise

async def periodic_maintenance():
    """Background task for cleaning expired sessions"""
    while True:
        try:
            await asyncio.sleep(1800)  # 30 minutes
            if memory_manager:
                cleaned = await memory_manager.cleanup_expired_sessions()
                if cleaned > 0:
                    print(f"🧹 Cleaned up {cleaned} expired sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Maintenance error: {e}")

# Dependency injection functions
async def get_memory_manager() -> ModernMemoryManager:
    """Dependency to get memory manager"""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    return memory_manager

async def get_llm_client() -> AzureChatOpenAI:
    """Dependency to get LLM client"""
    if langchain_chat_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    return langchain_chat_client

# FastAPI app
app = FastAPI(
    title="LangChain 1.0.x RAG API with In-Memory Storage",
    version="1.0.0",
    description="RAG chatbot with ConversationBufferWindowMemory using LangChain 1.0.x",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# API endpoints
@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    active_sessions = await memory_manager.get_active_sessions_count() if memory_manager else 0
    
    return {
        "status": "healthy",
        "service": "LangChain 1.0.x RAG API with Memory",
        "version": "1.0.0",
        "langchain_versions": {
            "langchain": "1.0.2",
            "langchain-openai": "1.0.1",
            "langchain-community": "0.4.1",
            "langchain-core": "1.0.1"
        },
        "memory_type": "ConversationBufferWindowMemory",
        "active_sessions": active_sessions,
        "features": [
            "LangChain 1.0.x compatibility",
            "ConversationBufferWindowMemory", 
            "In-memory session management",
            "Structured output parsing",
            "Async/await patterns"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

async def get_embeddings(query: str) -> List[float]:
    """Get embeddings using LangChain 1.0.x"""
    try:
        embeddings = await langchain_embeddings_client.aembed_query(query)
        return embeddings
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

async def search_documents(query: str) -> str:
    """Document search with Azure Cognitive Search"""
    try:
        query_vector = await get_embeddings(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=3,
            fields="text_vector"
        )

        results = await azure_search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=3,
            select=["chunk"]
        )

        context_chunks = []
        async for result in results:
            chunk = result.get("chunk", "")
            if chunk:
                context_chunks.append(chunk)
        
        return "\n\n".join(context_chunks)
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return ""

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    memory_manager: ModernMemoryManager = Depends(get_memory_manager),
    llm_client: AzureChatOpenAI = Depends(get_llm_client)
):
    """Chat endpoint with ConversationBufferWindowMemory"""
    try:
        # Generate or use provided session ID
        session_id = request.session_id or memory_manager.create_session()
        
        # If session doesn't exist, create it
        if request.session_id and await memory_manager.get_session_memory(request.session_id) is None:
            session_id = memory_manager.create_session(request.session_id)
        
        print(f"🔄 Processing chat - Session: {session_id}")
        
        # Get context from search
        context = await search_documents(request.query)
        
        # Create dynamic LLM with request parameters
        dynamic_llm = AzureChatOpenAI(
            azure_endpoint=llm_client.azure_endpoint,
            api_key=llm_client.api_key,
            api_version=llm_client.api_version,
            azure_deployment=llm_client.azure_deployment,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Create RAG chain with memory
        rag_chain = ModernMemoryRAGChain(
            llm=dynamic_llm,
            session_id=session_id,
            memory_manager=memory_manager
        )
        
        # Invoke chain
        result = await rag_chain.ainvoke({
            "input": request.query,
            "context": context
        })
        
        return ChatResponse(
            response=result["response"],
            followup_qs=result["followup_qs"],
            session_id=session_id,
            metadata={
                "context_length": len(context),
                "processing_time": datetime.utcnow().isoformat(),
                "memory_type": "ConversationBufferWindowMemory",
                "langchain_version": "1.0.2"
            }
        )
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/chat/history/{session_id}", response_model=ConversationHistoryResponse, tags=["Memory"])
async def get_conversation_history(
    session_id: str,
    limit: int = 20,
    memory_manager: ModernMemoryManager = Depends(get_memory_manager)
):
    """Get conversation history from ConversationBufferWindowMemory"""
    try:
        # Get session info
        session_info = await memory_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get chat history
        chat_history = await memory_manager.get_session_chat_history(session_id)
        if not chat_history:
            raise HTTPException(status_code=404, detail="Session history not found")
        
        # Get messages with limit
        messages = chat_history.messages
        if limit > 0:
            messages = messages[-limit:]
        
        formatted_messages = [
            {
                "type": type(msg).__name__,
                "content": msg.content,
                "timestamp": datetime.utcnow().isoformat()
            }
            for msg in messages
        ]
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=formatted_messages,
            total_messages=len(chat_history.messages),
            created_at=session_info.created_at.isoformat(),
            last_updated=session_info.last_updated.isoformat(),
            memory_window_size=session_info.memory_window_size,
            metadata={
                "retrieved_messages": len(formatted_messages),
                "limit_applied": limit,
                "memory_type": "ConversationBufferWindowMemory"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

@app.delete("/chat/history/{session_id}", tags=["Memory"])
async def clear_conversation_history(
    session_id: str,
    memory_manager: ModernMemoryManager = Depends(get_memory_manager)
):
    """Clear conversation history from memory"""
    try:
        success = await memory_manager.clear_session(session_id)
        
        if success:
            return {
                "message": f"Conversation history cleared for session {session_id}",
                "memory_type": "ConversationBufferWindowMemory",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History clearing failed: {str(e)}")

@app.get("/chat/sessions/stats", tags=["Analytics"])
async def get_session_stats(
    memory_manager: ModernMemoryManager = Depends(get_memory_manager)
):
    """Get comprehensive session statistics"""
    try:
        active_sessions = await memory_manager.get_active_sessions_count()
        session_list = await memory_manager.list_active_sessions()
        
        total_messages = sum(session.message_count for session in session_list)
        avg_messages = total_messages / len(session_list) if session_list else 0
        
        return {
            "statistics": {
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "average_messages_per_session": round(avg_messages, 2)
            },
            "memory_configuration": {
                "type": "ConversationBufferWindowMemory",
                "window_size": memory_manager.memory_window_size,
                "session_timeout_hours": memory_manager.session_timeout.total_seconds() / 3600,
                "storage": "In-Memory"
            },
            "system_info": {
                "langchain_versions": {
                    "langchain": "1.0.2",
                    "langchain-openai": "1.0.1",
                    "langchain-community": "0.4.1",
                    "langchain-core": "1.0.1"
                },
                "features": [
                    "LangChain 1.0.x compatibility",
                    "ConversationBufferWindowMemory",
                    "In-memory session management",
                    "Structured output parsing"
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")

@app.get("/chat/sessions/list", tags=["Analytics"])
async def list_active_sessions(
    memory_manager: ModernMemoryManager = Depends(get_memory_manager)
):
    """List all active sessions"""
    try:
        sessions = await memory_manager.list_active_sessions()
        
        return {
            "active_sessions": [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_updated": session.last_updated.isoformat(),
                    "message_count": session.message_count,
                    "memory_window_size": session.memory_window_size
                }
                for session in sessions
            ],
            "total_count": len(sessions),
            "memory_type": "ConversationBufferWindowMemory",
            "langchain_version": "1.0.2",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session listing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )











# Strands POC
from strands import Agent, tool
from strands_tools import calculator, current_time
from strands.models.openai import OpenAIModel
from dotenv import load_dotenv
import requests
import os


load_dotenv()

# Define a custom tool as a Python function using the @tool decorator
@tool
def get_weather(city: str) -> str:
    """
    Provide basic weather information of a given city.

    Args:
        city (str): The exact city to retrive the weather information

    Returns:
        str: Basic weather information for the city
    """
    print(f"🛠: Tool call, get_weather {city}")

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


model = OpenAIModel(
    client_args={"api_key": os.getenv("OPENAI_API_KEY")},
    model_id="gpt-4o-mini",  # Use your preferred OpenAI model name
    params={"temperature": 0.3},
)

WEATHER_SYSTEM_PROMPT = """
You are a helpful weather assistant.
When asked about the weather in a city, use the get_weather tool to fetch current weather data.
Always provide a clear and friendly summary.
"""


agent = Agent(system_prompt=WEATHER_SYSTEM_PROMPT, tools=[calculator, current_time, get_weather], model=model)

# Ask the agent a question that uses the available tools
message = """
I have 4 requests:

1. What is the time right now?
2. Calculate 3111696 / 74088
3. What is the weather in London?
"""
print(agent(message))



# Replace with your Azure OpenAI endpoint and API key
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_DEPLOYMENT_NAME = "your-deployment-name"  # e.g., "gpt-4o-mini"

# Initialize the model with Azure OpenAI settings
model = OpenAIModel(
    client_args={
        "api_key": AZURE_OPENAI_API_KEY,
        "base_url": f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}",
        "api_version": "2024-02-01",  # Use the latest Azure OpenAI API version
    },
    model_id=AZURE_OPENAI_DEPLOYMENT_NAME,
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)
https://github.com/strands-agents/sdk-python/discussions/1143





from strands.models.base import BaseModel
from openai import AzureOpenAI
import os

class AzureOpenAIModel(BaseModel):
    def __init__(self, api_key, azure_endpoint, api_version, deployment_id, **model_config):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version
        )
        self.deployment_id = deployment_id
        self.model_config = model_config

    def generate(self, prompt, **kwargs):
        params = {**self.model_config, **kwargs}
        response = self.client.chat.completions.create(
            model=self.deployment_id,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        return response.choices.message.content

    async def agenerate(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

from strands import Agent
from strands_tools import calculator

azure_model = AzureOpenAIModel(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-05-15",
    deployment_id="your-deployment-id",
    max_tokens=1000,
    temperature=0.7
)

agent = Agent(model=azure_model, tools=[calculator])

response = agent("What is 2+2?")
print(response)



https://learn.microsoft.com/en-us/answers/questions/5513522/supported-languages-for-azure-openai-gpt-4o
https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-models/concepts/models-sold-directly-by-azure?pivots=azure-openai&tabs=global-standard-aoai%2Cstandard-chat-completions%2Cglobal-standard#gpt-4








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
