import asyncio
import os
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import httpx
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Azure App Service endpoint
FASTAPI_BASE_URL = "https://<azure_app_service>.aseqa.abc.org"

server = Server("document-chat-mcp-server")

@server.list_tools()
async def list_tools():
    """List available tools for document chat and health check"""
    return [
        Tool(
            name="check_service_health",
            description="Check if the document chat service is running and healthy",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="chat_with_documents",
            description="Chat with official documents using the RAG system. Ask questions about document content and get AI-powered responses with follow-up suggestions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Your question or message about the documents"
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
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool calls for document chat and health check"""
    
    if name == "check_service_health":
        return await check_health()
    elif name == "chat_with_documents":
        return await chat_with_docs(arguments)
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def check_health():
    """Check service health status"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{FASTAPI_BASE_URL}/")
            response.raise_for_status()
            
            health_data = response.json()
            
            status_message = f"""
# Service Health Check ✅

**Status**: {health_data.get('status', 'unknown')}
**Message**: {health_data.get('message', 'No message')}
**Technology**: {health_data.get('using', 'Not specified')}
**Endpoint**: {FASTAPI_BASE_URL}

The document chat service is running and ready to answer questions!
            """.strip()
            
            return [TextContent(type="text", text=status_message)]
            
    except httpx.TimeoutException:
        error_msg = f"❌ **Service Timeout**: The service at {FASTAPI_BASE_URL} is not responding within 30 seconds."
        return [TextContent(type="text", text=error_msg)]
    except httpx.HTTPStatusError as e:
        error_msg = f"❌ **HTTP Error {e.response.status_code}**: Service returned an error - {e.response.text}"
        return [TextContent(type="text", text=error_msg)]
    except Exception as e:
        error_msg = f"❌ **Connection Error**: Cannot reach the service - {str(e)}"
        return [TextContent(type="text", text=error_msg)]

async def chat_with_docs(arguments: dict):
    """Chat with documents using the RAG system"""
    query = arguments.get("query", "").strip()
    max_tokens = arguments.get("max_tokens", 1000)
    temperature = arguments.get("temperature", 0.1)
    
    if not query:
        return [TextContent(
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
        
        logger.info(f"Sending chat request: {chat_request}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/chat",
                json=chat_request,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 400:
                error_detail = response.json().get("detail", "Bad request")
                return [TextContent(
                    type="text",
                    text=f"❌ **Invalid Request**: {error_detail}"
                )]
            elif response.status_code == 500:
                error_detail = response.json().get("detail", "Internal server error")
                return [TextContent(
                    type="text",
                    text=f"❌ **Server Error**: {error_detail}"
                )]
            
            response.raise_for_status()
            chat_data = response.json()
            
            # Format the response
            response_text = chat_data.get("response", "No response received")
            followup_questions = chat_data.get("followup_qs", [])
            
            formatted_response = f"""
# Document Chat Response 📚

**Your Question**: {query}

## Answer:
{response_text}
            """.strip()
            
            if followup_questions and len(followup_questions) > 0:
                formatted_response += "\n\n## Suggested Follow-up Questions:"
                for i, question in enumerate(followup_questions, 1):
                    formatted_response += f"\n{i}. {question}"
            
            formatted_response += f"\n\n---\n*Settings: max_tokens={max_tokens}, temperature={temperature}*"
            
            return [TextContent(type="text", text=formatted_response)]
            
    except httpx.TimeoutException:
        return [TextContent(
            type="text",
            text="❌ **Timeout Error**: The document chat request took too long to process (>60s). Please try a simpler question."
        )]
    except httpx.HTTPStatusError as e:
        error_msg = f"❌ **HTTP Error {e.response.status_code}**: {e.response.text}"
        return [TextContent(type="text", text=error_msg)]
    except json.JSONDecodeError:
        return [TextContent(
            type="text",
            text="❌ **Response Error**: Received invalid response from the service."
        )]
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_docs: {e}")
        return [TextContent(
            type="text",
            text=f"❌ **Unexpected Error**: {str(e)}"
        )]

async def main():
    """Main server entry point"""
    logger.info(f"Starting Document Chat MCP Server for {FASTAPI_BASE_URL}")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, 
            write_stream, 
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())




{
  "name": "document-chat-mcp-server",
  "version": "1.0.0",
  "description": "MCP server for Azure OpenAI RAG Document Chat API",
  "command": "python",
  "args": ["document_chat_mcp_server.py"],
  "env": {
    "PYTHONPATH": "."
  },
  "capabilities": [
    "tools"
  ],
  "metadata": {
    "fastapi_endpoint": "https://<azure_app_service>.aseqa.abc.org",
    "features": ["document_chat", "health_check", "rag_system"]
  }
}



mcp>=1.0.0
httpx>=0.25.0
asyncio
logging



from setuptools import setup, find_packages

setup(
    name="document-chat-mcp-server",
    version="1.0.0",
    description="MCP server for Azure OpenAI RAG Document Chat",
    py_modules=["document_chat_mcp_server"],
    install_requires=[
        "mcp>=1.0.0",
        "httpx>=0.25.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "document-chat-mcp-server=document_chat_mcp_server:main",
        ],
    },
    keywords=["mcp", "fastapi", "azure", "openai", "rag", "document-chat"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)



{
  "mcpServers": {
    "document-chat": {
      "command": "python",
      "args": ["path/to/document_chat_mcp_server.py"],
      "env": {}
    }
  }
}
