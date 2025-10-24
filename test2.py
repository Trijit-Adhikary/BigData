@app.on_event("startup")
async def startup_event():
    """Initialize Azure clients on startup"""
    global azure_model_client, azure_search_client
    
    try:
        # Initialize Azure OpenAI client
        azure_model_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY", "AZURE_API_KEY"),
            api_version="2025-01-01-preview",
            azure_endpoint=os.getenv("AZURE_ENDPOINT", "AZURE_END_POINT")
        )

        # Initialize Azure Search client
        azure_search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("INDEX_NAME", "INDEX_NAME"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY", "AZURE_SEARCH_ADMIN_KEY"))
        )
        
        print("✅ Azure clients initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing clients: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global azure_search_client
    if azure_search_client:
        await azure_search_client.close()
    print("🔄 Application shutdown complete")
