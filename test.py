pip install langchain langchain-openai langchain-community


# Request
{
    "query": "What is machine learning?",
    "max_tokens": 800,
    "temperature": 0.2
}

# Response
{
    "response": "Machine learning is a subset of artificial intelligence...",
    "followup_qs": [
        "What are the different types of machine learning algorithms?",
        "How do I choose the right machine learning model for my data?",
        "What are some real-world applications of machine learning?"
    ]
}
