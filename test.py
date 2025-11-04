from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime, timedelta
import asyncio

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# LangChain imports with MEMORY
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain_core.output_parsers import PydanticOutputParser
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

class SessionManager:
    """Manages multiple conversation sessions with LangChain memory"""
    
    def __init__(self, memory_window_size: int = 5, session_timeout_hours: int = 24):
        self.sessions: Dict[str, Dict] = {}
        self.memory_window_size = memory_window_size
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self._lock = asyncio.Lock()
    
    def create_session(self) -> str:
        """Create new session with LangChain memory"""
        session_id = str(uuid.uuid4())
        
        # Create LangChain ConversationBufferWindowMemory
        memory = ConversationBufferWindowMemory(
            k=self.memory_window_size,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        
        self.sessions[session_id] = {
            "memory": memory,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "conversation_count": 0
        }
        
        return session_id
    
    async def get_session_memory(self, session_id: str) -> Optional[ConversationBufferWindowMemory]:
        """Get LangChain memory for session"""
        async with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            
            # Check if session expired
            if datetime.now() - session["last_accessed"] > self.session_timeout:
                await self.cleanup_session(session_id)
                return None
            
            # Update last accessed
            session["last_accessed"] = datetime.now()
            return session["memory"]
    
    async def update_session(self, session_id: str, input_text: str, output_text: str):
        """Update session with new conversation turn"""
        async with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                # LangChain memory automatically handles the conversation history
                session["memory"].save_context(
                    {"input": input_text}, 
                    {"output": output_text}
                )
                session["conversation_count"] += 1
                session["last_accessed"] = datetime.now()
    
    async def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information"""
        async with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            memory = session["memory"]
            
            # Get conversation history from LangChain memory
            messages = memory.chat_memory.messages
            
            return {
                "session_id": session_id,
                "created_at": session["created_at"].isoformat(),
                "last_accessed": session["last_accessed"].isoformat(),
                "conversation_count": session["conversation_count"],
                "memory_buffer_length": len(messages),
                "messages": [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content,
                    }
                    for msg in messages
                ]
            }
    
    async def cleanup_session(self, session_id: str) -> bool:
        """Remove session"""
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
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
    
    async def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        await self.cleanup_expired_sessions()
        return len(self.sessions)

# Global clients and session manager
langchain_chat_client = None
langchain_embeddings_client = None
azure_search_client = None
session_manager = SessionManager(memory_window_size=5, session_timeout_hours=24)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients and background tasks"""
    global langchain_chat_client, langchain_embeddings_client, azure_search_client
    
    try:
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
        
        print("✅ LangChain clients with memory initialized successfully")
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        raise

    yield
    
    # Cancel cleanup task
    if 'cleanup_task' in locals():
        cleanup_task.cancel()
    
    if azure_search_client:
        await azure_search_client.close()
    print("🔄 Application shutdown complete")

async def periodic_cleanup():
    """Periodic cleanup using LangChain memory management"""
    while True:
        try:
            await asyncio.sleep(1800)  # Run every 30 minutes
            cleaned = await session_manager.cleanup_expired_sessions()
            if cleaned > 0:
                print(f"🧹 LangChain Memory: Cleaned up {cleaned} expired sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Memory cleanup error: {e}")

app = FastAPI(
    title="LangChain RAG API with Native Memory",
    version="0.127.0",
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

class RAGConversationChain:
    """Custom RAG chain using LangChain memory"""
    
    def __init__(self, llm, memory: ConversationBufferWindowMemory):
        self.llm = llm
        self.memory = memory
        
        # Create custom prompt template with memory placeholder
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a helpful AI assistant with conversation memory. Answer the user's question based on the provided context and conversation history.

Current Context from Documents:
{context}

Instructions:
- Use the conversation history to provide contextually aware responses
- Reference previous questions/answers when relevant
- Provide a clear, concise answer to the current question
- Generate exactly 3 follow-up questions
- Return response in JSON format like this:
{{"response": "your answer", "followup_qs": ["q1", "q2", "q3"]}}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        # Create parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=LangChainResponse)
    
    async def invoke(self, input_text: str, context: str) -> dict:
        """Invoke the RAG chain with memory"""
        try:
            # Get chat history from memory
            memory_variables = self.memory.load_memory_variables({})
            chat_history = memory_variables.get("chat_history", [])
            
            # Format the prompt
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
                # Fallback parsing
                result = await self._fallback_parse(response.content, input_text)
            
            return result
            
        except Exception as e:
            print(f"❌ RAG Chain error: {e}")
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

async def generate_response_with_langchain_memory(
    user_query: str, 
    session_id: str, 
    context: str, 
    max_tokens: int, 
    temperature: float
):
    """Generate response using LangChain's native memory"""
    try:
        # Get or create session memory
        memory = await session_manager.get_session_memory(session_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Create dynamic LLM with custom parameters
        dynamic_llm = AzureChatOpenAI(
            azure_endpoint=langchain_chat_client.azure_endpoint,
            api_key=langchain_chat_client.api_key,
            api_version=langchain_chat_client.api_version,
            azure_deployment=langchain_chat_client.azure_deployment,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create RAG chain with LangChain memory
        rag_chain = RAGConversationChain(dynamic_llm, memory)
        
        # Generate response
        result = await rag_chain.invoke(user_query, context)
        
        # Update session memory (LangChain handles this automatically in the chain)
        await session_manager.update_session(session_id, user_query, result["response"])
        
        return result
        
    except Exception as e:
        print(f"❌ LangChain memory response error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/")
async def health_check():
    """Health check with LangChain memory stats"""
    active_sessions = await session_manager.get_active_sessions_count()
    
    return {
        "status": "healthy",
        "message": "LangChain RAG API with Native Memory",
        "memory_type": "ConversationBufferWindowMemory",
        "active_sessions": active_sessions,
        "memory_window_size": session_manager.memory_window_size
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using LangChain's native memory"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get or create session ID
        session_id = request.session_id or session_manager.create_session()
        
        print(f"🔄 Processing with LangChain Memory - Query: {request.query[:50]}... | Session: {session_id}")
        
        # Get context from search
        context = await search_documents(request.query)
        print(f"📄 Context found: {len(context)} characters")
        
        # Generate response with LangChain memory
        result = await generate_response_with_langchain_memory(
            request.query,
            session_id,
            context,
            request.max_tokens,
            request.temperature
        )
        
        print(f"✅ LangChain Memory Response generated for session: {session_id}")
        
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

@app.get("/chat/memory/{session_id}")
async def get_session_memory_info(session_id: str):
    """Get LangChain memory information for session"""
    try:
        session_info = await session_manager.get_session_info(session_id)
        
        if session_info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Clear LangChain memory for session"""
    try:
        success = await session_manager.cleanup_session(session_id)
        
        if success:
            return {"message": f"LangChain memory cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/memory/stats")
async def get_memory_stats():
    """Get overall LangChain memory statistics"""
    active_sessions = await session_manager.get_active_sessions_count()
    
    return {
        "memory_type": "ConversationBufferWindowMemory",
        "window_size": session_manager.memory_window_size,
        "active_sessions": active_sessions,
        "session_timeout_hours": session_manager.session_timeout.total_seconds() / 3600,
        "timestamp": datetime.now().isoformat()
    }
