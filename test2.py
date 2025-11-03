from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime, timedelta
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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

class ConversationTurn(BaseModel):
    """Single Q&A turn in conversation"""
    question: str
    answer: str
    timestamp: datetime

class AsyncConversationMemory:
    """Fully asynchronous conversation memory with thread-safe operations"""
    
    def __init__(self, max_turns: int = 5, session_timeout_hours: int = 24):
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.session_timestamps: Dict[str, datetime] = {}
        self.max_turns = max_turns
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self._lock = asyncio.Lock()  # Async lock for thread safety
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def create_session(self) -> str:
        """Async session creation"""
        async with self._lock:
            session_id = str(uuid.uuid4())
            self.conversations[session_id] = []
            self.session_timestamps[session_id] = datetime.now()
            await asyncio.sleep(0)  # Yield control
            return session_id
    
    async def add_turn(self, session_id: str, question: str, answer: str):
        """Async add Q&A turn to conversation memory"""
        async with self._lock:
            if session_id not in self.conversations:
                self.conversations[session_id] = []
            
            turn = ConversationTurn(
                question=question,
                answer=answer,
                timestamp=datetime.now()
            )
            
            self.conversations[session_id].append(turn)
            
            # Keep only last N turns
            if len(self.conversations[session_id]) > self.max_turns:
                self.conversations[session_id] = self.conversations[session_id][-self.max_turns:]
            
            # Update session timestamp
            self.session_timestamps[session_id] = datetime.now()
            await asyncio.sleep(0)  # Yield control
    
    async def get_conversation_history(self, session_id: str) -> List[ConversationTurn]:
        """Async get conversation history for session"""
        async with self._lock:
            if session_id not in self.conversations:
                return []
            
            # Check if session has expired
            if await self._is_session_expired(session_id):
                await self._cleanup_session(session_id)
                return []
            
            await asyncio.sleep(0)  # Yield control
            return self.conversations[session_id].copy()
    
    async def _is_session_expired(self, session_id: str) -> bool:
        """Async check if session has expired"""
        if session_id not in self.session_timestamps:
            return True
        
        await asyncio.sleep(0)  # Yield control
        return datetime.now() - self.session_timestamps[session_id] > self.session_timeout
    
    async def _cleanup_session(self, session_id: str):
        """Async remove expired session"""
        self.conversations.pop(session_id, None)
        self.session_timestamps.pop(session_id, None)
        await asyncio.sleep(0)  # Yield control
    
    async def cleanup_expired_sessions(self):
        """Async cleanup all expired sessions"""
        async with self._lock:
            expired_sessions = []
            
            for sid in list(self.session_timestamps.keys()):
                if await self._is_session_expired(sid):
                    expired_sessions.append(sid)
            
            for session_id in expired_sessions:
                await self._cleanup_session(session_id)
            
            await asyncio.sleep(0)  # Yield control
            return len(expired_sessions)
    
    async def get_active_sessions_count(self) -> int:
        """Async get count of active sessions"""
        await self.cleanup_expired_sessions()
        async with self._lock:
            count = len(self.conversations)
            await asyncio.sleep(0)  # Yield control
            return count
    
    async def clear_session(self, session_id: str) -> bool:
        """Async clear specific session"""
        async with self._lock:
            existed = session_id in self.conversations
            self.conversations.pop(session_id, None)
            self.session_timestamps.pop(session_id, None)
            await asyncio.sleep(0)  # Yield control
            return existed

# Global clients and memory
langchain_chat_client = None
langchain_embeddings_client = None
azure_search_client = None
conversation_memory = AsyncConversationMemory(max_turns=5, session_timeout_hours=24)

async def init_langchain_clients():
    """Async initialization of LangChain clients"""
    global langchain_chat_client, langchain_embeddings_client
    
    # Initialize chat client asynchronously
    langchain_chat_client = AzureChatOpenAI(
        azure_endpoint="AZURE_OPENAI_ENDPOINT",
        api_key="AZURE_OPENAI_KEY",
        api_version="2024-06-01",
        azure_deployment="gpt-4o",
        temperature=0.1
    )

    # Initialize embeddings client asynchronously
    langchain_embeddings_client = AzureOpenAIEmbeddings(
        azure_endpoint="AZURE_OPENAI_ENDPOINT", 
        api_key="AZURE_OPENAI_KEY",
        api_version="2024-06-01",
        azure_deployment="text-embedding-ada-002"
    )
    
    await asyncio.sleep(0)  # Yield control
    return langchain_chat_client, langchain_embeddings_client

async def init_azure_search_client():
    """Async initialization of Azure Search client"""
    global azure_search_client
    
    azure_search_client = SearchClient(
        endpoint="AZURE_SEARCH_ENDPOINT",
        index_name="rag-hellopdf", 
        credential=AzureKeyCredential("AZURE_SEARCH_ADMIN_KEY")
    )
    
    await asyncio.sleep(0)  # Yield control
    return azure_search_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan with proper resource management"""
    global langchain_chat_client, langchain_embeddings_client, azure_search_client
    
    try:
        print("🚀 Initializing async clients...")
        
        # Initialize all clients concurrently
        chat_task = asyncio.create_task(init_langchain_clients())
        search_task = asyncio.create_task(init_azure_search_client())
        
        # Wait for both to complete
        await asyncio.gather(chat_task, search_task)
        
        print("✅ All async clients initialized successfully")
        
        # Start background tasks
        cleanup_task = asyncio.create_task(periodic_cleanup())
        health_monitor_task = asyncio.create_task(periodic_health_check())
        
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        raise

    yield
    
    # Cleanup tasks
    print("🔄 Starting graceful shutdown...")
    
    cleanup_task.cancel()
    health_monitor_task.cancel()
    
    # Wait for tasks to finish
    await asyncio.gather(cleanup_task, health_monitor_task, return_exceptions=True)
    
    if azure_search_client:
        await azure_search_client.close()
    
    print("✅ Application shutdown complete")

async def periodic_cleanup():
    """Async periodic cleanup of expired sessions"""
    while True:
        try:
            await asyncio.sleep(1800)  # Run every 30 minutes
            cleaned = await conversation_memory.cleanup_expired_sessions()
            if cleaned > 0:
                print(f"🧹 Cleaned up {cleaned} expired sessions")
        except asyncio.CancelledError:
            print("🔄 Cleanup task cancelled")
            break
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
            await asyncio.sleep(60)  # Wait before retrying

async def periodic_health_check():
    """Async periodic health monitoring"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            active_sessions = await conversation_memory.get_active_sessions_count()
            print(f"💚 Health check: {active_sessions} active sessions")
        except asyncio.CancelledError:
            print("🔄 Health monitor task cancelled")
            break
        except Exception as e:
            print(f"❌ Health check error: {e}")
            await asyncio.sleep(60)

app = FastAPI(
    title="Fully Async LangChain RAG API with Memory",
    version="0.126.0",
    lifespan=lifespan
)

async def get_embeddings_async(query: str):
    """Fully async embeddings generation"""
    try:
        # Run embeddings in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            conversation_memory.executor,
            langchain_embeddings_client.embed_query,
            query
        )
        return embeddings
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

async def search_documents_async(query: str):
    """Async document search with concurrent processing"""
    try:
        # Get embeddings asynchronously
        query_vector = await get_embeddings_async(query)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=3,
            fields="text_vector"
        )

        # Perform async search
        results = await azure_search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=3,
            select=["chunk"]
        )

        # Process results asynchronously
        context_chunks = []
        async for result in results:
            chunk = result.get("chunk", "")
            if chunk:
                context_chunks.append(chunk)
            await asyncio.sleep(0)  # Yield control during iteration
        
        return "\n".join(context_chunks)
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return ""

async def build_conversation_context_async(session_id: str) -> str:
    """Async build conversation context from memory"""
    history = await conversation_memory.get_conversation_history(session_id)
    
    if not history:
        return ""
    
    context_parts = ["Previous conversation:"]
    
    for i, turn in enumerate(history, 1):
        context_parts.append(f"Q{i}: {turn.question}")
        context_parts.append(f"A{i}: {turn.answer}")
        await asyncio.sleep(0)  # Yield control during iteration
    
    context_parts.append("---")
    return "\n".join(context_parts)

async def generate_response_async(user_query: str, session_id: str, context: str, max_tokens: int, temperature: float):
    """Fully async response generation with memory"""
    try:
        # Get conversation history asynchronously
        conversation_context = await build_conversation_context_async(session_id)
        
        # Build system message with context and memory
        system_content = f"""
You are a helpful AI assistant with conversation memory. Answer the user's question based on the provided context and conversation history.

{conversation_context}

Current Context from Documents:
{context}

Instructions:
- Use the conversation history to provide contextually aware responses
- Reference previous questions/answers when relevant
- Provide a clear, concise answer to the current question
- Generate exactly 3 follow-up questions
- Return response in JSON format like this:
{{"response": "your answer", "followup_qs": ["q1", "q2", "q3"]}}
"""

        # Create messages
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_query)
        ]

        # Create client with dynamic parameters
        dynamic_client = AzureChatOpenAI(
            azure_endpoint=langchain_chat_client.azure_endpoint,
            api_key=langchain_chat_client.api_key, 
            api_version=langchain_chat_client.api_version,
            azure_deployment=langchain_chat_client.azure_deployment,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Generate response asynchronously
        response = await dynamic_client.ainvoke(messages)
        
        # Parse response asynchronously
        content = response.content.strip()
        parsed_result = await parse_response_async(content)
        
        # Store in conversation memory asynchronously
        await conversation_memory.add_turn(session_id, user_query, parsed_result["response"])
        
        return parsed_result
        
    except Exception as e:
        print(f"❌ Response generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

async def parse_response_async(content: str) -> dict:
    """Async response parsing"""
    try:
        # Remove code blocks if present
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
        
        # Parse JSON asynchronously (using executor for CPU-bound task)
        loop = asyncio.get_event_loop()
        parsed = await loop.run_in_executor(
            conversation_memory.executor,
            json.loads,
            content
        )
        
        if "response" in parsed and "followup_qs" in parsed:
            followup_qs = parsed["followup_qs"][:3]
            while len(followup_qs) < 3:
                followup_qs.append("Can you tell me more?")
            
            return {
                "response": parsed["response"],
                "followup_qs": followup_qs
            }
    
    except (json.JSONDecodeError, Exception):
        pass
    
    # Fallback response
    return {
        "response": content,
        "followup_qs": [
            "What are the main aspects?",
            "How does this work in practice?",
            "Can you provide examples?"
        ]
    }

@app.get("/")
async def health_check():
    """Async health check with memory stats"""
    active_sessions = await conversation_memory.get_active_sessions_count()
    
    return {
        "status": "healthy",
        "message": "Fully Async LangChain RAG API with Memory",
        "active_sessions": active_sessions,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint_async(request: ChatRequest):
    """Fully async main chat endpoint with memory"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get or create session ID asynchronously
        session_id = request.session_id or await conversation_memory.create_session()
        
        print(f"🔄 Processing query: {request.query[:50]}... | Session: {session_id}")
        
        # Run search and response generation concurrently
        search_task = asyncio.create_task(search_documents_async(request.query))
        
        # Wait for search to complete
        context = await search_task
        print(f"📄 Context found: {len(context)} characters")
        
        # Generate response with memory
        result = await generate_response_async(
            request.query,
            session_id,
            context,
            request.max_tokens,
            request.temperature
        )
        
        print(f"✅ Response generated for session: {session_id}")
        
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

@app.get("/chat/history/{session_id}")
async def get_conversation_history_async(session_id: str):
    """Async get conversation history for a session"""
    try:
        history = await conversation_memory.get_conversation_history(session_id)
        
        return {
            "session_id": session_id,
            "conversation_count": len(history),
            "history": [
                {
                    "question": turn.question,
                    "answer": turn.answer,
                    "timestamp": turn.timestamp.isoformat()
                }
                for turn in history
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/history/{session_id}")
async def clear_conversation_history_async(session_id: str):
    """Async clear conversation history for a session"""
    try:
        existed = await conversation_memory.clear_session(session_id)
        
        if existed:
            return {"message": f"Conversation history cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/active")
async def get_active_sessions_async():
    """Async get count of active sessions"""
    active_count = await conversation_memory.get_active_sessions_count()
    total_conversations = sum(len(conv) for conv in conversation_memory.conversations.values())
    
    return {
        "active_sessions": active_count,
        "total_conversations": total_conversations,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat/batch")
async def batch_chat_async(requests: List[ChatRequest]):
    """Async batch processing of multiple chat requests"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")
    
    try:
        # Process all requests concurrently
        tasks = [
            chat_endpoint_async(request) 
            for request in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses.append({
                    "error": str(result),
                    "request_index": i
                })
            else:
                responses.append(result.dict())
        
        return {"results": responses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
