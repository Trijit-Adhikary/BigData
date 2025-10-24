@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for vector search-powered conversations
    
    - **query**: User's question or message
    - **max_tokens**: Maximum tokens in response (default: 500)
    - **temperature**: Response creativity (0.0-1.0, default: 0.1)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response_text, chunk_count = await generate_chat_response(
            request.query, 
            request.max_tokens, 
            request.temperature
        )
        
        return ChatResponse(
            response=response_text,
            context_chunks_count=chunk_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
