from azure.search.documents.models import VectorizedQuery

def query_vectorizer(query: str, azure_model_client, embd_model: str):
    response = azure_model_client.embeddings.create(
        input=query,
        model=embd_model
    )
    return response.data[0].embedding


def vector_search(query: str, azure_model_client, azure_search_client, embd_model: str):
    query_vector = query_vectorizer(query, azure_model_client, embd_model)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=5,
        fields="text_vector"
    )

    results = azure_search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        query_type="semantic",
        semantic_configuration_name="rag-hellopdf-semantic-configuration",
        top=5,
        select=["chunk"]
    )

    return results


def chat(azure_model_client, azure_search_client, embd_mode):
    user_query = input("> ")

    current_context = ""
    messages = []

    sys_prompt = f"""

    You are a helpful AI assistent who understands the user query carefully and then answers the question based on the current context.

    current context:
    {current_context}

    Keep your answers short and precise
    Ask follow up questions to the user to help the user dig deeper

    """

    chunks = vector_search(user_query, azure_model_client, azure_search_client, embd_mode)
    current_context = chunks

    messages.append({"role":"system", "content": sys_prompt})
    messages.append({"role":"user", "content": user_query})

    # Generate the completion
    completion = azure_model_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500,
        temperature=0.1,
        top_p=0.95
    )

    print(completion.choices[0].message.content)
