from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import uuid
from datetime import datetime, timedelta
import asyncio
import os
from dotenv import load_dotenv

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# Modern LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.runnable import Runnable
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Modern Pydantic models with Field validation
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User's chat query")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    max_tokens: Optional[int] = Field(800, ge=100, le=2000, description="Maximum tokens for response")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Temperature for response generation")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is machine learning?",
                "session_id": "optional-session-id",
                "max_tokens": 800,
                "temperature": 0.1
            }
        }
    }

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI assistant response")
    followup_qs: List[str] = Field(..., description="Follow-up questions")
    session_id: str = Field(..., description="Session identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class StructuredResponse(BaseModel):
    """Structured response model for LLM output parsing"""
    response: str = Field(..., description="The main response to the user's query")
    followup_qs: List[str] = Field(..., min_length=3, max_length=3, description="Exactly 3 follow-up questions")

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

# Modern in-memory session manager with ConversationBufferWindowMemory
class ModernMemoryManager:
    """Modern session manager using ConversationBufferWindowMemory"""
    
    def __init__(self, memory_window_size: int = 5, session_timeout_hours: int = 24):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.memory_window_size = memory_window_size
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self._lock = asyncio.Lock()
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create new session with modern ConversationBufferWindowMemory"""
        session_id = session_id or str(uuid.uuid4())
        
        # Create modern ChatMessageHistory for the session
        chat_history = ChatMessageHistory()
        
        # Create ConversationBufferWindowMemory with modern configuration
        memory = ConversationBufferWindowMemory(
            k=self.memory_window_size,
            chat_memory=chat_history,
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True,  # Return as message objects instead of strings
            human_prefix="User",
            ai_prefix="Assistant"
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
            memory = session["memory"]
            
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

# Modern callback handler for logging
class ConversationCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for conversation logging"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = None
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        self.start_time = datetime.now()
        print(f"🔄 Chain started for session {self.session_id}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"✅ Chain completed for session {self.session_id} in {duration.total_seconds():.2f}s")
    
    def on_chain_error(self, error: Exception, **kwargs) -> None:
        print(f"❌ Chain error for session {self.session_id}: {error}")

# Modern LangChain Chain with ConversationBufferWindowMemory
class ModernMemoryRAGChain:
    """Modern LangChain chain using ConversationBufferWindowMemory and LCEL"""
    
    def __init__(self, llm: AzureChatOpenAI, session_id: str, memory_manager: ModernMemoryManager):
        self.llm = llm
        self.session_id = session_id
        self.memory_manager = memory_manager
        
        # Modern prompt template with better structure
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an intelligent AI assistant with conversation memory. 
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
            HumanMessage(content="{input}")
        ])
        
        # Modern output parser
        self.output_parser = PydanticOutputParser(pydantic_object=StructuredResponse)
    
    async def ainvoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Modern async invocation with ConversationBufferWindowMemory"""
        try:
            # Get memory for the session
            memory = await self.memory_manager.get_session_memory(self.session_id)
            if memory is None:
                raise HTTPException(status_code=404, detail="Session not found or expired")
            
            # Get chat history from memory
            memory_variables = memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            # Create the chain using LCEL
            chain = (
                RunnablePassthrough.assign(
                    format_instructions=lambda _: self.output_parser.get_format_instructions()
                )
                | self.prompt
                | self.llm
                | self.output_parser
            )
            
            # Add callback handler
            callback_handler = ConversationCallbackHandler(self.session_id)
            
            # Invoke chain
            result = await chain.ainvoke(
                {
                    "input": input_data["input"],
                    "context": input_data.get("context", ""),
                    "chat_history": chat_history
                },
                config={"callbacks": [callback_handler]}
            )
            
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
            print(f"❌ Modern memory chain invocation error: {e}")
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
langchain_chat_client = None
langchain_embeddings_client = None
azure_search_client = None
memory_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan management with in-memory storage"""
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
        
        # Initialize modern LangChain clients
        langchain_chat_client = AzureChatOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version="2024-06-01",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            temperature=0.1,
            streaming=True,  # Enable streaming
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
        
        print("✅ Modern LangChain application with memory initialized successfully")
        
        # Start background tasks
        cleanup_task = asyncio.create_task(periodic_maintenance())
        
        yield
        
        # Graceful shutdown
        print("🔄 Starting graceful shutdown...")
        cleanup_task.cancel()
        
        # Close resources
        if azure_search_client:
            await azure_search_client.close()
        
        print("✅ Modern application shutdown completed")
        
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

# Modern dependency injection
async def get_memory_manager() -> ModernMemoryManager:
    """Dependency to get memory manager"""
    return memory_manager

async def get_llm_client() -> AzureChatOpenAI:
    """Dependency to get LLM client"""
    return langchain_chat_client

# Modern FastAPI app
app = FastAPI(
    title="Modern LangChain RAG API with In-Memory Storage",
    version="1.0.0",
    description="Advanced RAG chatbot with ConversationBufferWindowMemory using latest LangChain patterns",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Modern API endpoints
@app.get("/", tags=["Health"])
async def health_check():
    """Modern health check endpoint"""
    active_sessions = await memory_manager.get_active_sessions_count() if memory_manager else 0
    
    return {
        "status": "healthy",
        "service": "Modern LangChain RAG API with Memory",
        "version": "1.0.0",
        "langchain_version": "0.1.5",
        "memory_type": "ConversationBufferWindowMemory",
        "active_sessions": active_sessions,
        "features": [
            "Modern LCEL chains",
            "ConversationBufferWindowMemory", 
            "In-memory session management",
            "Structured output parsing",
            "Async/await patterns"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

async def get_embeddings(query: str) -> List[float]:
    """Modern embeddings with error handling"""
    try:
        embeddings = await langchain_embeddings_client.aembed_query(query)
        return embeddings
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

async def search_documents(query: str) -> str:
    """Modern document search with optimization"""
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
async def modern_memory_chat_endpoint(
    request: ChatRequest,
    memory_manager: ModernMemoryManager = Depends(get_memory_manager),
    llm_client: AzureChatOpenAI = Depends(get_llm_client)
):
    """Modern chat endpoint with ConversationBufferWindowMemory"""
    try:
        # Generate or use provided session ID
        session_id = request.session_id or memory_manager.create_session()
        
        # If session doesn't exist, create it
        if request.session_id and await memory_manager.get_session_memory(request.session_id) is None:
            session_id = memory_manager.create_session(request.session_id)
        
        print(f"🔄 Processing modern memory chat - Session: {session_id}")
        
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
        
        # Create modern RAG chain with memory
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
                "memory_type": "ConversationBufferWindowMemory"
            }
        )
        
    except Exception as e:
        print(f"❌ Modern memory chat error: {e}")
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
                "langchain_version": "0.1.5",
                "features": [
                    "Modern LCEL chains",
                    "ConversationBufferWindowMemory",
                    "In-memory session management",
                    "Structured output parsing",
                    "Real-time analytics"
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
