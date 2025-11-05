from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime, timedelta
import asyncio
import os

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# LangChain imports with SQL Memory
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser

# SQL Database imports
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, Text, Integer, select
from sqlalchemy.dialects.postgresql import UUID
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_tokens: Optional[int] = 800
    temperature: Optional[float] = 0.1

class ChatResponse(BaseModel):
    response: str
    followup_qs: List[str]
    session_id: str

class LangChainResponse(BaseModel):
    response: str
    followup_qs: List[str]

class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict]
    total_messages: int
    created_at: str
    last_updated: str

# Database Models
Base = declarative_base()

class ConversationSession(Base):
    """Table to track conversation sessions"""
    __tablename__ = "conversation_sessions"
    
    session_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_messages = Column(Integer, default=0)
    user_info = Column(Text, nullable=True)  # For future user tracking

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://username:password@localhost/chatbot_db"
)

class AsyncSQLChatMessageHistory:
    """Async wrapper for LangChain SQL Chat Message History"""
    
    def __init__(self, session_id: str, connection_string: str):
        self.session_id = session_id
        self.connection_string = connection_string
        self._sync_history = None
    
    def _get_sync_history(self):
        """Get synchronous SQLChatMessageHistory instance"""
        if self._sync_history is None:
            # Convert async connection string to sync for LangChain
            sync_connection_string = self.connection_string.replace("+asyncpg", "")
            self._sync_history = SQLChatMessageHistory(
                session_id=self.session_id,
                connection_string=sync_connection_string,
                table_name="message_store"
            )
        return self._sync_history
    
    async def add_message(self, message: BaseMessage):
        """Add message to SQL storage"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self._get_sync_history().add_message, 
                message
            )
        except Exception as e:
            print(f"❌ Error adding message to SQL: {e}")
            raise
    
    async def get_messages(self, limit: int = 5) -> List[BaseMessage]:
        """Get latest messages from SQL storage"""
        try:
            loop = asyncio.get_event_loop()
            all_messages = await loop.run_in_executor(
                None, 
                self._get_sync_history().messages.copy
            )
            
            # Return latest 'limit' messages
            return all_messages[-limit:] if len(all_messages) > limit else all_messages
        
        except Exception as e:
            print(f"❌ Error retrieving messages from SQL: {e}")
            return []
    
    async def clear(self):
        """Clear messages for this session"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self._get_sync_history().clear
            )
        except Exception as e:
            print(f"❌ Error clearing messages from SQL: {e}")
            raise

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def init_db(self):
        """Initialize database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            raise
    
    async def create_or_update_session(self, session_id: str):
        """Create or update conversation session"""
        try:
            async with self.async_session() as session:
                # Check if session exists
                result = await session.execute(
                    select(ConversationSession).where(
                        ConversationSession.session_id == session_id
                    )
                )
                existing_session = result.scalar_one_or_none()
                
                if existing_session:
                    # Update existing session
                    existing_session.last_updated = datetime.utcnow()
                    existing_session.total_messages += 1
                else:
                    # Create new session
                    new_session = ConversationSession(
                        session_id=session_id,
                        total_messages=1
                    )
                    session.add(new_session)
                
                await session.commit()
                
        except Exception as e:
            print(f"❌ Error creating/updating session: {e}")
            raise
    
    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ConversationSession).where(
                        ConversationSession.session_id == session_id
                    )
                )
                session_info = result.scalar_one_or_none()
                
                if session_info:
                    return {
                        "session_id": session_info.session_id,
                        "created_at": session_info.created_at.isoformat(),
                        "last_updated": session_info.last_updated.isoformat(),
                        "total_messages": session_info.total_messages
                    }
                return None
                
        except Exception as e:
            print(f"❌ Error getting session info: {e}")
            return None
    
    async def get_active_sessions_count(self, hours: int = 24) -> int:
        """Get count of active sessions in last N hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            async with self.async_session() as session:
                result = await session.execute(
                    select(ConversationSession).where(
                        ConversationSession.last_updated >= cutoff_time
                    )
                )
                sessions = result.scalars().all()
                return len(sessions)
                
        except Exception as e:
            print(f"❌ Error getting active sessions count: {e}")
            return 0
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session and its messages"""
        try:
            async with self.async_session() as session:
                # Delete session record
                result = await session.execute(
                    select(ConversationSession).where(
                        ConversationSession.session_id == session_id
                    )
                )
                session_record = result.scalar_one_or_none()
                
                if session_record:
                    await session.delete(session_record)
                    await session.commit()
                    
                    # Also clear messages from LangChain's message store
                    sql_history = AsyncSQLChatMessageHistory(session_id, self.database_url)
                    await sql_history.clear()
                    
                    return True
                return False
                
        except Exception as e:
            print(f"❌ Error deleting session: {e}")
            return False

# Global clients and managers
langchain_chat_client = None
langchain_embeddings_client = None
azure_search_client = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients, database, and background tasks"""
    global langchain_chat_client, langchain_embeddings_client, azure_search_client, db_manager
    
    try:
        # Initialize Database Manager
        db_manager = DatabaseManager(DATABASE_URL)
        await db_manager.init_db()
        
        # Initialize LangChain clients
        langchain_chat_client = AzureChatOpenAI(
            azure_endpoint="AZURE_OPENAI_ENDPOINT",
            api_key="AZURE_OPENAI_KEY",
            api_version="2024-06-01",
            azure_deployment="gpt-4o",
            temperature=0.1
        )

        langchain_embeddings_client = AzureOpenAIEmbeddings(
            azure_endpoint="AZURE_OPENAI_ENDPOINT", 
            api_key="AZURE_OPENAI_KEY",
            api_version="2024-06-01",
            azure_deployment="text-embedding-ada-002"
        )

        azure_search_client = SearchClient(
            endpoint="AZURE_SEARCH_ENDPOINT",
            index_name="rag-hellopdf", 
            credential=AzureKeyCredential("AZURE_SEARCH_ADMIN_KEY")
        )
        
        print("✅ All clients and SQL database initialized successfully")
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
    except Exception as e:
        print(f"❌ Error initializing application: {e}")
        raise

    yield
    
    # Cancel cleanup task
    if 'cleanup_task' in locals():
        cleanup_task.cancel()
    
    if azure_search_client:
        await azure_search_client.close()
    
    if db_manager:
        await db_manager.engine.dispose()
    
    print("🔄 Application shutdown complete")

async def periodic_cleanup():
    """Periodic cleanup of old sessions"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            # Could implement cleanup of very old sessions here
            active_count = await db_manager.get_active_sessions_count(24)
            print(f"💚 Health check: {active_count} active sessions in last 24h")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

app = FastAPI(
    title="LangChain RAG API with SQL Memory",
    version="0.128.0",
    lifespan=lifespan
)

async def get_embeddings(query: str):
    """Get embeddings using LangChain"""
    try:
        embeddings = langchain_embeddings_client.embed_query(query)
        return embeddings
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

async def search_documents(query: str):
    """Search documents using vector similarity"""
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
        
        return "\n".join(context_chunks)
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return ""

class SQLRAGChain:
    """RAG Chain with SQL-backed conversation memory"""
    
    def __init__(self, llm, session_id: str, database_url: str):
        self.llm = llm
        self.session_id = session_id
        self.sql_history = AsyncSQLChatMessageHistory(session_id, database_url)
        
        # Create prompt template with message history
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a helpful AI assistant with persistent conversation memory stored in a database. 
Answer the user's question based on the provided context and conversation history.

Current Context from Documents:
{context}

Instructions:
- Use the conversation history to provide contextually aware responses
- Reference previous questions/answers when relevant
- Maintain conversation continuity across sessions
- Provide a clear, concise answer to the current question
- Generate exactly 3 follow-up questions
- Return response in JSON format like this:
{{"response": "your answer", "followup_qs": ["q1", "q2", "q3"]}}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        # Parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=LangChainResponse)
    
    async def invoke(self, input_text: str, context: str) -> dict:
        """Invoke RAG chain with SQL memory"""
        try:
            # Get latest 5 messages from SQL database
            chat_history = await self.sql_history.get_messages(limit=10)  # Get 10 to have 5 pairs
            
            # Format the prompt with history and context
            formatted_prompt = self.prompt.format_messages(
                context=context,
                chat_history=chat_history,
                input=input_text
            )
            
            # Generate response
            response = await self.llm.ainvoke(formatted_prompt)
            
            # Parse structured response
            try:
                parsed_response = self.parser.parse(response.content)
                result = {
                    "response": parsed_response.response,
                    "followup_qs": parsed_response.followup_qs
                }
            except Exception as parse_error:
                print(f"⚠️ Parsing error, using fallback: {parse_error}")
                result = await self._fallback_parse(response.content, input_text)
            
            # Store conversation in SQL database
            await self.sql_history.add_message(HumanMessage(content=input_text))
            await self.sql_history.add_message(AIMessage(content=result["response"]))
            
            # Update session info
            await db_manager.create_or_update_session(self.session_id)
            
            return result
            
        except Exception as e:
            print(f"❌ SQL RAG Chain error: {e}")
            raise HTTPException(status_code=500, detail=f"Chain execution failed: {str(e)}")
    
    async def _fallback_parse(self, content: str, input_text: str) -> dict:
        """Fallback response parsing"""
        try:
            # Try to extract JSON
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            parsed = json.loads(content)
            
            if "response" in parsed and "followup_qs" in parsed:
                followup_qs = parsed["followup_qs"][:3]
                while len(followup_qs) < 3:
                    followup_qs.append(f"Can you tell me more about {input_text}?")
                
                return {
                    "response": parsed["response"],
                    "followup_qs": followup_qs
                }
        except json.JSONDecodeError:
            pass
        
        # Final fallback
        return {
            "response": content,
            "followup_qs": [
                f"What are the key aspects of {input_text}?",
                f"How does {input_text} work in practice?",
                f"Can you provide examples of {input_text}?"
            ]
        }

async def generate_response_with_sql_memory(
    user_query: str, 
    session_id: str, 
    context: str, 
    max_tokens: int, 
    temperature: float
):
    """Generate response using SQL-backed LangChain memory"""
    try:
        # Create dynamic LLM with custom parameters
        dynamic_llm = AzureChatOpenAI(
            azure_endpoint=langchain_chat_client.azure_endpoint,
            api_key=langchain_chat_client.api_key,
            api_version=langchain_chat_client.api_version,
            azure_deployment=langchain_chat_client.azure_deployment,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create SQL RAG chain
        sql_rag_chain = SQLRAGChain(dynamic_llm, session_id, DATABASE_URL)
        
        # Generate response with SQL memory
        result = await sql_rag_chain.invoke(user_query, context)
        
        return result
        
    except Exception as e:
        print(f"❌ SQL memory response error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/")
async def health_check():
    """Health check with SQL memory stats"""
    active_sessions = await db_manager.get_active_sessions_count(24)
    
    return {
        "status": "healthy",
        "message": "LangChain RAG API with SQL Memory",
        "memory_type": "SQLChatMessageHistory",
        "database_url": DATABASE_URL.split("@")[1] if "@" in DATABASE_URL else "Not configured",
        "active_sessions_24h": active_sessions
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using SQL-backed LangChain memory"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get or create session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        print(f"🔄 Processing with SQL Memory - Query: {request.query[:50]}... | Session: {session_id}")
        
        # Get context from search
        context = await search_documents(request.query)
        print(f"📄 Context found: {len(context)} characters")
        
        # Generate response with SQL memory
        result = await generate_response_with_sql_memory(
            request.query,
            session_id,
            context,
            request.max_tokens,
            request.temperature
        )
        
        print(f"✅ SQL Memory Response generated for session: {session_id}")
        
        return ChatResponse(
            response=result["response"],
            followup_qs=result["followup_qs"],
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/chat/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, limit: int = 10):
    """Get conversation history from SQL database"""
    try:
        # Get session info
        session_info = await db_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get messages from SQL
        sql_history = AsyncSQLChatMessageHistory(session_id, DATABASE_URL)
        messages = await sql_history.get_messages(limit=limit)
        
        # Format messages
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "type": type(msg).__name__,
                "content": msg.content,
                "timestamp": datetime.utcnow().isoformat()  # Could be enhanced with actual timestamps
            })
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=formatted_messages,
            total_messages=session_info["total_messages"],
            created_at=session_info["created_at"],
            last_updated=session_info["last_updated"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history from SQL database"""
    try:
        success = await db_manager.delete_session(session_id)
        
        if success:
            return {"message": f"SQL conversation history cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/stats")
async def get_memory_stats():
    """Get SQL memory statistics"""
    active_sessions_1h = await db_manager.get_active_sessions_count(1)
    active_sessions_24h = await db_manager.get_active_sessions_count(24)
    active_sessions_7d = await db_manager.get_active_sessions_count(168)  # 7 days
    
    return {
        "memory_type": "SQLChatMessageHistory + PostgreSQL",
        "database_url": DATABASE_URL.split("@")[1] if "@" in DATABASE_URL else "Not configured",
        "active_sessions": {
            "last_1_hour": active_sessions_1h,
            "last_24_hours": active_sessions_24h,
            "last_7_days": active_sessions_7d
        },
        "features": [
            "Persistent conversation storage",
            "Latest 5 message pairs for context",
            "Cross-session conversation continuity",
            "SQL-based message retrieval"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/chat/sessions/list")
async def list_active_sessions(hours: int = 24):
    """List active sessions (admin endpoint)"""
    try:
        # This would require additional database queries
        # For now, return count and basic info
        active_count = await db_manager.get_active_sessions_count(hours)
        
        return {
            "active_sessions_count": active_count,
            "time_window_hours": hours,
            "note": "Detailed session listing can be implemented based on requirements"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
