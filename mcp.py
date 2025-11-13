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


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    memory_manager: LangchainMemoryManager = Depends(get_memory_manager),
    llm_client: AzureChatOpenAI = Depends(get_llm_client)
):
    """Chat endpoint with ConversationBufferWindowMemory"""
    try:
        # Handle session ID: create new or use existing
        if request.session_id:
            # User provided a session_id - check if it exists
            existing_memory = await memory_manager.get_session_memory(request.session_id)
            if existing_memory is None:
                # Session doesn't exist, create it with the provided ID
                session_id = memory_manager.create_session(request.session_id)
            else:
                # Session exists, use it
                session_id = request.session_id
        else:
            # No session_id provided - create a new one
            session_id = memory_manager.create_session()
        
        print(f"🔄 Processing chat - Session: {session_id}")
        
        # Get context from search
        context = await search_documents(request.query)
        
        # Create dynamic LLM with request parameters
        # dynamic_llm = AzureChatOpenAI(
        #     azure_endpoint=llm_client.azure_endpoint,
        #     api_key=llm_client.api_key,
        #     api_version=llm_client.api_version,
        #     azure_deployment=llm_client.azure_deployment,
        #     temperature=request.temperature,
        #     max_tokens=request.max_tokens
        # )
        
        dynamic_llm=langchain_chat_client
        # Create RAG chain with memory
        rag_chain = LangchainMemoryRAGChain(
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
