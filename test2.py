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

# SQL Database imports - AZURE SQL COMPATIBLE
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, Text, Integer, select, NVARCHAR
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER  # Azure SQL specific
import pyodbc  # For Azure SQL
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

# Database Models - AZURE SQL COMPATIBLE
Base = declarative_base()

class ConversationSession(Base):
    """Table to track conversation sessions - Azure SQL Compatible"""
    __tablename__ = "conversation_sessions"
    
    session_id = Column(NVARCHAR(50), primary_key=True, index=True)  # Changed to NVARCHAR
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_messages = Column(Integer, default=0)
    user_info = Column(Text, nullable=True)

# Azure SQL Database configuration
AZURE_SQL_CONNECTION_STRING = os.getenv(
    "AZURE_SQL_CONNECTION_STRING",
    "mssql+aiodbc://username:password@server.database.windows.net/database?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)

class AzureSQLChatMessageHistory:
    """Azure SQL compatible wrapper for LangChain Chat Message History"""
    
    def __init__(self, session_id: str, connection_string: str):
        self.session_id = session_id
        self.connection_string = connection_string
        self._sync_history = None
    
    def _get_sync_history(self):
        """Get synchronous SQLChatMessageHistory instance for Azure SQL"""
        if self._sync_history is None:
            # Convert async connection string to sync for LangChain
            sync_connection_string = self.connection_string.replace("+aiodbc", "+pyodbc")
            
            self._sync_history = SQLChatMessageHistory(
                session_id=self.session_id,
                connection_string=sync_connection_string,
                table_name="message_store",
                # Azure SQL specific configurations
                custom_message_converter=self._azure_sql_message_converter
            )
        return self._sync_history
    
    def _azure_sql_message_converter(self, message: BaseMessage) -> str:
        """Convert message to Azure SQL compatible JSON string"""
        return json.dumps({
            "type": type(message).__name__,
            "content": message.content,
            "additional_kwargs": getattr(message, 'additional_kwargs', {})
        })
    
    async def add_message(self, message: BaseMessage):
        """Add message to Azure SQL storage"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                self._get_sync_history().add_message, 
                message
            )
        except Exception as e:
            print(f"❌ Error adding message to Azure SQL: {e}")
            raise
    
    async def get_messages(self, limit: int = 5) -> List[BaseMessage]:
        """Get latest messages from Azure SQL storage"""
        try:
            loop = asyncio.get_event_loop()
            all_messages = await loop.run_in_executor(
                None, 
                self._get_sync_history().messages.copy
            )
            
            # Return latest 'limit' messages
            return all_messages[-limit:] if len(all_messages) > limit else all_messages
        
        except Exception as e:
            print(f"❌ Error retrieving messages from Azure SQL: {e}")
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
            print(f"❌ Error clearing messages from Azure SQL: {e}")
            raise

class AzureDatabaseManager:
    """Azure SQL Database manager"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
        # Azure SQL specific engine configuration
        self.engine = create_async_engine(
            connection_string,
            echo=False,
            # Azure SQL specific configurations
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
                "autocommit": False
            }
        )
        
        self.async_session = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def init_db(self):
        """Initialize Azure SQL database tables"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("✅ Azure SQL Database tables created successfully")
            
            # Create indexes for better performance
            await self._create_indexes()
            
        except Exception as e:
            print(f"❌ Azure SQL Database initialization error: {e}")
            raise
    
    async def _create_indexes(self):
        """Create Azure SQL specific indexes"""
        try:
            async with self.engine.begin() as conn:
                # Create index on session_id for message_store table
                await conn.execute(text("""
                    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_message_store_session_id')
                    CREATE INDEX IX_message_store_session_id ON message_store(session_id)
                """))
                
                # Create index on created_at for better date queries
                await conn.execute(text("""
                    IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_conversation_sessions_created_at')
                    CREATE INDEX IX_conversation_sessions_created_at ON conversation_sessions(created_at)
                """))
                
        except Exception as e:
            print(f"⚠️ Warning: Could not create indexes: {e}")
    
    async def create_or_update_session(self, session_id: str):
        """Create or update conversation session in Azure SQL"""
        try:
            async with self.async_session() as session:
                # Use Azure SQL compatible query
                stmt = select(ConversationSession).where(
                    ConversationSession.session_id == session_id
                )
                result = await session.execute(stmt)
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
            print(f"❌ Error creating/updating session in Azure SQL: {e}")
            raise
    
    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information from Azure SQL"""
        try:
            async with self.async_session() as session:
                stmt = select(ConversationSession).where(
                    ConversationSession.session_id == session_id
                )
                result = await session.execute(stmt)
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
            print(f"❌ Error getting session info from Azure SQL: {e}")
            return None
    
    async def get_active_sessions_count(self, hours: int = 24) -> int:
        """Get count of active sessions in last N hours from Azure SQL"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            async with self.async_session() as session:
                stmt = select(ConversationSession).where(
                    ConversationSession.last_updated >= cutoff_time
                )
                result = await session.execute(stmt)
                sessions = result.scalars().all()
                return len(sessions)
                
        except Exception as e:
            print(f"❌ Error getting active sessions count from Azure SQL: {e}")
            return 0
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session and its messages from Azure SQL"""
        try:
            async with self.async_session() as session:
                # Delete session record
                stmt = select(ConversationSession).where(
                    ConversationSession.session_id == session_id
                )
                result = await session.execute(stmt)
                session_record = result.scalar_one_or_none()
                
                if session_record:
                    await session.delete(session_record)
                    await session.commit()
                    
                    # Also clear messages from LangChain's message store
                    sql_history = AzureSQLChatMessageHistory(session_id, self.connection_string)
                    await sql_history.clear()
                    
                    return True
                return False
                
        except Exception as e:
            print(f"❌ Error deleting session from Azure SQL: {e}")
            return False

# Global clients and managers
langchain_chat_client = None
langchain_embeddings_client = None
azure_search_client = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients, Azure SQL database, and background tasks"""
    global langchain_chat_client, langchain_embeddings_client, azure_search_client, db_manager
    
    try:
        # Initialize Azure SQL Database Manager
        db_manager = AzureDatabaseManager(AZURE_SQL_CONNECTION_STRING)
        await db_manager.init_db()
        
        # Initialize LangChain clients
        langchain_chat_client = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-06-01",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            temperature=0.1
        )

        langchain_embeddings_client = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-06-01",
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        )

        azure_search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX", "rag-hellopdf"), 
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        )
        
        print("✅ All clients and Azure SQL database initialized successfully")
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
    except Exception as e:
        print(f"❌ Error initializing application with Azure SQL: {e}")
        raise

    yield
    
    # Cancel cleanup task
    if 'cleanup_task' in locals():
        cleanup_task.cancel()
    
    if azure_search_client:
        await azure_search_client.close()
    
    if db_manager:
        await db_manager.engine.dispose()
    
    print("🔄 Application with Azure SQL shutdown complete")

# [Rest of the code remains the same with AzureSQLChatMessageHistory instead of AsyncSQLChatMessageHistory]

class AzureSQLRAGChain:
    """RAG Chain with Azure SQL-backed conversation memory"""
    
    def __init__(self, llm, session_id: str, connection_string: str):
        self.llm = llm
        self.session_id = session_id
        self.sql_history = AzureSQLChatMessageHistory(session_id, connection_string)
        
        # Create prompt template with message history
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a helpful AI assistant with persistent conversation memory stored in Azure SQL Database. 
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
        """Invoke RAG chain with Azure SQL memory"""
        try:
            # Get latest 5 messages from Azure SQL database
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
            
            # Store conversation in Azure SQL database
            await self.sql_history.add_message(HumanMessage(content=input_text))
            await self.sql_history.add_message(AIMessage(content=result["response"]))
            
            # Update session info
            await db_manager.create_or_update_session(self.session_id)
            
            return result
            
        except Exception as e:
            print(f"❌ Azure SQL RAG Chain error: {e}")
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

# [Continue with the rest of the FastAPI endpoints using AzureSQLRAGChain and AzureDatabaseManager]

app = FastAPI(
    title="LangChain RAG API with Azure SQL Memory",
    version="0.129.0",
    lifespan=lifespan
)

# [All the other endpoints remain the same, just replace the class names]
