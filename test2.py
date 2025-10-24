async def generate_chat_response(user_query: str, max_tokens: int, temperature: float):
    """Generate chat response using Azure OpenAI"""
    try:
        # Get relevant chunks from vector search
        chunks = await vector_search(user_query, azure_model_client, azure_search_client, embd_model)
        
        # Convert search results to context string
        context_chunks = []
        chunk_count = 0
        async for chunk in chunks:
            context_chunks.append(chunk.get("chunk", ""))
            chunk_count += 1
        
        current_context = "\n".join(context_chunks)

        sys_prompt = f"""
        You are a helpful AI assistant who understands the user query carefully and then answers the question based on the current context.

        current context:
        {current_context}

        Keep your answers short and precise
        Ask follow up questions to the user to help the user dig deeper
        """

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_query}
        ]

        # Generate the completion asynchronously
        completion = await azure_model_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95
        )

        return completion.choices[0].message.content, chunk_count

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
