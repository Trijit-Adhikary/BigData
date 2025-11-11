import asyncio
import json
import logging
from typing import Any, Sequence
import httpx

from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Azure App Service endpoint
FASTAPI_BASE_URL = "https://<azure_app_service>.aseqa.abc.org"

# Create the server instance
server = mcp.server.Server("document-chat-mcp-server")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for document chat and health check"""
    return [
        types.Tool(
            name="check_service_health",
            description="Check if the document chat service is running and healthy",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="chat_with_documents", 
            description="Chat with official documents using the RAG system. Ask questions about document content and get AI-powered responses with follow-up suggestions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Your question or message about the documents",
                        "minLength": 1
                    },
                    "max_tokens": {
                        "type": "integer", 
                        "description": "Maximum tokens in response (default: 1000)",
                        "default": 1000,
                        "minimum": 100,
                        "maximum": 4000
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Response creativity level (0.0-1.0, default: 0.1)", 
                        "default": 0.1,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any] | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls for document chat and health check"""
    
    if arguments is None:
        arguments = {}
    
    match name:
        case "check_service_health":
            return await check_health()
        case "chat_with_documents":
            return await chat_with_docs(arguments)
        case _:
            raise ValueError(f"Unknown tool: {name}")

async def check_health() -> list[types.TextContent]:
    """Check service health status"""
    try:
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{FASTAPI_BASE_URL}/")
            response.raise_for_status()
            
            health_data = response.json()
            
            status_message = f"""# Service Health Check ✅

**Status**: {health_data.get('status', 'unknown')}
**Message**: {health_data.get('message', 'No message')}
**Technology**: {health_data.get('using', 'Not specified')}
**Endpoint**: {FASTAPI_BASE_URL}

The document chat service is running and ready to answer questions!"""
            
            return [types.TextContent(type="text", text=status_message)]
            
    except httpx.TimeoutException:
        error_msg = f"❌ **Service Timeout**: The service at {FASTAPI_BASE_URL} is not responding within 30 seconds."
        return [types.TextContent(type="text", text=error_msg)]
    except httpx.HTTPStatusError as e:
        error_msg = f"❌ **HTTP Error {e.response.status_code}**: Service returned an error"
        if e.response.text:
            error_msg += f" - {e.response.text[:200]}"
        return [types.TextContent(type="text", text=error_msg)]
    except Exception as e:
        logger.error(f"Health check error: {e}")
        error_msg = f"❌ **Connection Error**: Cannot reach the service - {str(e)}"
        return [types.TextContent(type="text", text=error_msg)]

async def chat_with_docs(arguments: dict[str, Any]) -> list[types.TextContent]:
    """Chat with documents using the RAG system"""
    query = arguments.get("query", "").strip()
    max_tokens = arguments.get("max_tokens", 1000)
    temperature = arguments.get("temperature", 0.1)
    
    if not query:
        return [types.TextContent(
            type="text",
            text="❌ **Error**: Query cannot be empty. Please provide a question about the documents."
        )]
    
    try:
        # Prepare the chat request
        chat_request = {
            "query": query,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        logger.info(f"Sending chat request: query='{query[:50]}...', max_tokens={max_tokens}, temp={temperature}")
        
        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/chat",
                json=chat_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                error_data = response.json() if response.content else {"detail": "Bad request"}
                error_detail = error_data.get("detail", "Bad request")
                return [types.TextContent(
                    type="text",
                    text=f"❌ **Invalid Request**: {error_detail}"
                )]
            elif response.status_code == 500:
                error_data = response.json() if response.content else {"detail": "Internal server error"}
                error_detail = error_data.get("detail", "Internal server error")
                return [types.TextContent(
                    type="text",
                    text=f"❌ **Server Error**: {error_detail}"
                )]
            
            response.raise_for_status()
            chat_data = response.json()
            
            # Format the response
            response_text = chat_data.get("response", "No response received")
            followup_questions = chat_data.get("followup_qs", [])
            
            formatted_response = f"""# Document Chat Response 📚

**Your Question**: {query}

## Answer:
{response_text}"""
            
            if followup_questions and len(followup_questions) > 0:
                formatted_response += "\n\n## Suggested Follow-up Questions:"
                for i, question in enumerate(followup_questions, 1):
                    formatted_response += f"\n{i}. {question}"
            
            formatted_response += f"\n\n---\n*Settings: max_tokens={max_tokens}, temperature={temperature}*"
            
            return [types.TextContent(type="text", text=formatted_response)]
            
    except httpx.TimeoutException:
        return [types.TextContent(
            type="text",
            text="❌ **Timeout Error**: The document chat request took too long to process (>60s). Please try a simpler question."
        )]
    except httpx.HTTPStatusError as e:
        error_msg = f"❌ **HTTP Error {e.response.status_code}**: Request failed"
        return [types.TextContent(type="text", text=error_msg)]
    except json.JSONDecodeError:
        return [types.TextContent(
            type="text",
            text="❌ **Response Error**: Received invalid JSON response from the service."
        )]
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_docs: {e}")
        return [types.TextContent(
            type="text",
            text=f"❌ **Unexpected Error**: {str(e)}"
        )]

async def main():
    """Main server entry point using latest MCP patterns"""
    logger.info(f"Starting Document Chat MCP Server for {FASTAPI_BASE_URL}")
    
    # Use the current standard stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="document-chat-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                )
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())



# requirements.txt - Latest MCP SDK
mcp>=0.9.0
httpx>=0.25.0


import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_document_chat():
    """Example usage with latest MCP client patterns"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["document_chat_mcp_server.py"],
        env=None,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            init_result = await session.initialize()
            logger.info(f"Server initialized: {init_result}")
            
            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
            
            # Check health
            health_result = await session.call_tool("check_service_health", {})
            print(f"Health check: {health_result.content[0].text}")
            
            # Chat with documents
            chat_result = await session.call_tool(
                "chat_with_documents",
                {
                    "query": "What are the main policies described in the official documents?",
                    "max_tokens": 800,
                    "temperature": 0.2
                }
            )
            print(f"Chat response: {chat_result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(use_document_chat())





{
  "mcpServers": {
    "document-chat": {
      "command": "python",
      "args": ["document_chat_mcp_server.py"],
      "env": {}
    }
  }
}
