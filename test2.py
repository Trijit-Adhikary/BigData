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

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # Optional session ID
    max_tokens: Optional[int] = 800
    temperature: Optional[float] = 0.1

class ChatResponse(BaseModel):
    response: str
    followup_qs: List[str]
    session_id: str  # Return session ID to client

class ConversationTurn(BaseModel):
    """Single Q&A turn in conversation"""
    question: str
    answer: str
    timestamp: datetime

class ConversationMemory:
    """In-memory conversation storage"""
    def __init__(self, max_turns: int = 5, session_timeout_hours: int = 24):
        self.conversations: Dict[str, List[ConversationTurn]] = {}
        self.session_timestamps: Dict[str, datetime] = {}
        self.max_turns = max_turns
        self.session_timeout = timedelta(hours=session_timeout_hours)
    
    def create_session(self) -> str:
        """Create new session ID"""
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        self.session_timestamps[session_id] = datetime.now()
        return session_id
    
    def add_turn(self, session_id: str, question: str, answer: str):
        """Add Q&A turn to conversation memory"""
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
    
    def get_conversation_history(self, session_id: str) -> List[ConversationTurn]:
        """Get conversation history for session"""
        if session_id not in self.conversations:
            return []
        
        # Check if session has expired
        if self._is_session_expired(session_id):
            self._cleanup_session(session_id)
            return []
        
        return self.conversations[session_id]
    
    def _is_session_expired(self, session_id: str) -> bool:
        """Check if session has expired"""
        if session_id not in self.session_timestamps:
            return True
        
        return datetime.now() - self.session_timestamps[session_id] > self.session_timeout
    
    def _cleanup_session(self, session_id: str):
        """Remove expired session"""
        self.conversations.pop(session_id, None)
        self.session_timestamps.pop(session_id, None)
    
    def cleanup_expired_sessions(self):
        """Clean up all expired sessions"""
        expired_sessions = [
            sid for sid in self.session_timestamps.keys()
            if self._is_session_expired(sid)
        ]
        
        for session_id in expired_sessions:
            self._cleanup_session(session_id)
        
        return len(expired_sessions)
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        self.cleanup_expired_sessions()
        return len(self.conversations)

# Global clients and memory
langchain_chat_client = None
langchain_embeddings_client = None
azure_search_client = None
conversation_memory = ConversationMemory(max_turns=5, session_timeout_hours=24)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients and cleanup task"""
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
        
        print("✅ All clients initialized successfully")
        
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
    """Periodic cleanup of expired sessions"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            cleaned = conversation_memory.cleanup_expired_sessions()
            if cleaned > 0:
                print(f"🧹 Cleaned up {cleaned} expired sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Cleanup error: {e}")

app = FastAPI(
    title="LangChain RAG API with Memory",
    version="0.125.0",
    lifespan=lifespan
)

async def get_embeddings(query: str):
    """Get embeddings using LangChain"""
    try:
        embeddings = langchain_embeddings_client.embed_query(query)
        return embeddings
    except Exception as e:
        print(f"Embedding error: {e}")
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
        print(f"Search error: {e}")
        return ""

def build_conversation_context(session_id: str) -> str:
    """Build conversation context from memory"""
    history = conversation_memory.get_conversation_history(session_id)
    
    if not history:
        return ""
    
    context_parts = ["Previous conversation:"]
    
    for i, turn in enumerate(history, 1):
        context_parts.append(f"Q{i}: {turn.question}")
        context_parts.append(f"A{i}: {turn.answer}")
    
    context_parts.append("---")
    return "\n".join(context_parts)

async def generate_response_with_memory(user_query: str, session_id: str, context: str, max_tokens: int, temperature: float):
    """Generate response using conversation memory"""
    try:
        # Get conversation history
        conversation_context = build_conversation_context(session_id)
        
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

        # Generate response
        response = await dynamic_client.ainvoke(messages)
        
        # Parse response
        content = response.content.strip()
        
        try:
            # Remove code blocks if present
            if content.startswith('```json'):
                content = content[7:-3].strip()
            elif content.startswith('```'):
                content = content[3:-3].strip()
            
            parsed = json.loads(content)
            
            if "response" in parsed and "followup_qs" in parsed:
                followup_qs = parsed["followup_qs"][:3]
                while len(followup_qs) < 3:
                    followup_qs.append(f"Can you tell me more about {user_query}?")
                
                result = {
                    "response": parsed["response"],
                    "followup_qs": followup_qs
                }
                
                # Store in conversation memory
                conversation_memory.add_turn(session_id, user_query, parsed["response"])
                
                return result
        
        except json.JSONDecodeError:
            pass
        
        # Fallback response
        result = {
            "response": content,
            "followup_qs": [
                f"What are the main aspects of {user_query}?",
                f"How does {user_query} work in practice?",
                f"What are some examples of {user_query}?"
            ]
        }
        
        # Store in conversation memory
        conversation_memory.add_turn(session_id, user_query, content)
        
        return result
        
    except Exception as e:
        print(f"Response generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.get("/")
async def health_check():
    """Health check with memory stats"""
    return {
        "status": "healthy",
        "message": "LangChain RAG API with Memory",
        "active_sessions": conversation_memory.get_active_sessions_count()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with memory"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get or create session ID
        session_id = request.session_id or conversation_memory.create_session()
        
        print(f"Processing query: {request.query} | Session: {session_id}")
        
        # Get context from search
        context = await search_documents(request.query)
        print(f"Context found: {len(context)} characters")
        
        # Generate response with memory
        result = await generate_response_with_memory(
            request.query,
            session_id,
            context,
            request.max_tokens,
            request.temperature
        )
        
        return ChatResponse(
            response=result["response"],
            followup_qs=result["followup_qs"],
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/chat/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = conversation_memory.get_conversation_history(session_id)
        
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
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        if session_id in conversation_memory.conversations:
            del conversation_memory.conversations[session_id]
            del conversation_memory.session_timestamps[session_id]
            return {"message": f"Conversation history cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/active")
async def get_active_sessions():
    """Get count of active sessions"""
    return {
        "active_sessions": conversation_memory.get_active_sessions_count(),
        "total_conversations": sum(len(conv) for conv in conversation_memory.conversations.values())
    }
