from azure.search.documents.aio import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI, AsyncAzureOpenAI
import aioconsole
import asyncio

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

    async with azure_search_client:
        results = await azure_search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name="rag-hellopdf-semantic-configuration",
            top=5,
            select=["chunk"]
        )

    return results


async def chat(azure_model_client, azure_search_client, embd_mode):
    user_query = await aioconsole.ainput("> ")

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
    azure_model_client = AsyncAzureOpenAI(
        api_key="AZURE_API_KEY",
        api_version="2025-01-01-preview",
        azure_endpoint="AZURE_END_POINT"
    )

    azure_search_client = SearchClient(
        endpoint="AZURE_SEARCH_ENDPOINT",
        index_name="INDEX_NAME",
        credential=AzureKeyCredential("AZURE_SEARCH_ADMIN_KEY")
    )

    embd_mode = "text-embedding-ada-002"

    await chat(azure_model_client, azure_search_client, embd_mode)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
