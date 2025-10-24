from azure.search.documents.models import VectorizedQuery

async def query_vectorizer(query: str, azure_model_client, embd_model: str):
    response = await azure_model_client.embeddings.create(
        input=query,
        model=embd_model
    )
    return response.data[0].embedding

async def vector_search(query: str, azure_model_client, azure_search_client, embd_model: str):
    query_vector = await query_vectorizer(query, azure_model_client, embd_model)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=5,
        fields="text_vector"
    )

    results = await azure_search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="rag-hellopdf-semantic-configuration",
        top=5,
        select=["chunk"]
    )

    return results

import asyncio

async def chat(azure_model_client, azure_search_client, embd_mode):
    # For async input, you might want to use a different approach
    # Since input() is blocking, consider using aiofiles or similar
    user_query = input("> ")  # Keep as is for now, or implement async input

    current_context = ""
    messages = []

    sys_prompt = f"""
    You are a helpful AI assistant who understands the user query carefully and then answers the question based on the current context.

    current context:
    {current_context}

    Keep your answers short and precise
    Ask follow up questions to the user to help the user dig deeper
    """

    chunks = await vector_search(user_query, azure_model_client, azure_search_client, embd_mode)
    current_context = chunks

    messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_query})

    # Generate the completion asynchronously
    completion = await azure_model_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,
        temperature=0.1,
        top_p=0.95
    )

    print(completion.choices[0].message.content)


# To run the async chat function
async def main():
    # Initialize your clients here
    await chat(azure_model_client, azure_search_client, embd_mode)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())



import aioconsole
user_query = await aioconsole.ainput("> ")
