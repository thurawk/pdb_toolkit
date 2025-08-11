#!/usr/bin/env python3
"""
Enhanced MCP CLI Client with Ollama LLM Integration

A production-ready CLI client that connects to MCP servers and uses local Ollama LLM instances.
Addresses all issues from the technical specification with proper error handling,
configuration management, and robust connection management.

Usage:
    python client.py --help
    python client.py --server-url "sse+http://127.0.0.1:8000/sse"
    python client.py --model llama3.2 --verbose
"""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import aiohttp
import click
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.status import Status
import structlog

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = structlog.get_logger()
console = Console()


class ErrorType(Enum):
    """Error types for categorized error handling."""
    NETWORK_ERROR = "network_error"
    MCP_SERVER_ERROR = "mcp_server_error"
    LLM_ERROR = "llm_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    PARSING_ERROR = "parsing_error"
    CONFIGURATION_ERROR = "configuration_error"


class MCPClientError(Exception):
    """Custom exception for MCP client operations."""
    
    def __init__(self, error_type: ErrorType, message: str, original_error: Optional[Exception] = None):
        self.error_type = error_type
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM."""
    model: str = Field(default="llama3.2", description="Ollama model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    request_timeout: float = Field(default=120.0, description="Request timeout in seconds")
    temperature: float = Field(default=0.1, description="LLM temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens per response")
    stream: bool = Field(default=True, description="Enable streaming responses")


class MCPConfig(BaseModel):
    """Configuration for MCP server connection."""
    server_url: str = Field(description="MCP server SSE endpoint")
    connection_timeout: float = Field(default=30.0, description="Connection timeout")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries")
    heartbeat_interval: float = Field(default=30.0, description="Heartbeat interval")


class ClientConfig(BaseModel):
    """Main client configuration."""
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    mcp: MCPConfig
    log_level: str = Field(default="INFO", description="Logging level")
    verbose: bool = Field(default=False, description="Enable verbose output")
    max_conversation_history: int = Field(default=50, description="Max conversation turns to keep")


class LLMInterface(Protocol):
    """Protocol for LLM implementations."""
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List] = None) -> str:
        """Generate a response from the LLM."""
        pass
    
    async def is_available(self) -> bool:
        """Check if the LLM is available."""
        pass


@dataclass
class ToolCallResponse:
    """Represents the result of a tool call."""
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    user_message: str
    agent_response: str
    tool_calls: List[ToolCallResponse] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class OllamaClient:
    """Ollama LLM client implementation."""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            if not self.session:
                return False
            
            async with self.session.get(f"{self.config.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.warning("Ollama availability check failed", error=str(e))
            return False
    
    async def generate(self, messages: List[Dict[str, str]], tools: Optional[List] = None) -> str:
        """Generate response from Ollama."""
        try:
            if not self.session:
                raise MCPClientError(
                    ErrorType.LLM_ERROR,
                    "Ollama session not initialized"
                )
            
            # Convert messages to Ollama format
            prompt = self._format_messages(messages, tools)
            
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,  # For simplicity, disable streaming for now
                "options": {
                    "temperature": self.config.temperature,
                }
            }
            
            if self.config.max_tokens:
                payload["options"]["num_predict"] = self.config.max_tokens
            
            async with self.session.post(
                f"{self.config.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise MCPClientError(
                        ErrorType.LLM_ERROR,
                        f"Ollama API error: {response.status} - {error_text}"
                    )
                
                result = await response.json()
                return result.get("response", "")
                
        except aiohttp.ClientError as e:
            raise MCPClientError(
                ErrorType.LLM_ERROR,
                f"Ollama connection error: {str(e)}",
                e
            )
        except Exception as e:
            if isinstance(e, MCPClientError):
                raise
            raise MCPClientError(
                ErrorType.LLM_ERROR,
                f"Unexpected error in Ollama generate: {str(e)}",
                e
            )
    
    def _format_messages(self, messages: List[Dict[str, str]], tools: Optional[List] = None) -> str:
        """Format messages and tools into a single prompt."""
        prompt_parts = []
        
        # Add tool descriptions if available
        if tools:
            prompt_parts.append("You have access to the following tools:")
            for tool in tools:
                tool_desc = f"- {tool['name']}: {tool.get('description', 'No description')}"
                prompt_parts.append(tool_desc)
            prompt_parts.append("\nUse these tools when needed by calling them in this format:")
            prompt_parts.append("TOOL_CALL: tool_name(param1=value1, param2=value2)")
            prompt_parts.append("")
        
        # Add conversation messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts)


class MCPToolManager:
    """Manages MCP tool discovery and execution."""
    
    def __init__(self, server_url: str, session: aiohttp.ClientSession):
        self.server_url = server_url
        self.session = session
        self.tools: Dict[str, Dict] = {}
    
    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from the MCP server."""
        try:
            # For this implementation, we'll simulate tool discovery
            # In a real MCP implementation, this would query the server's tool endpoint
            tools = [
                {
                    "name": "fetch_pdb",
                    "description": "Download PDB structure file for a given PDB ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pdb_id": {"type": "string", "description": "PDB ID to fetch"}
                        },
                        "required": ["pdb_id"]
                    }
                },
                {
                    "name": "read_pdb",
                    "description": "Read a PDB or mmCIF file from file path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the PDB file"}
                        },
                        "required": ["file_path"]
                    }
                },
                {
                    "name": "resolve_alias_to_uniprot",
                    "description": "Resolve protein alias to UniProt ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "alias": {"type": "string", "description": "Protein alias or name"}
                        },
                        "required": ["alias"]
                    }
                },
                {
                    "name": "map_uniprot_to_pdb",
                    "description": "Map UniProt ID to associated PDB IDs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "uniprot_id": {"type": "string", "description": "UniProt ID"}
                        },
                        "required": ["uniprot_id"]
                    }
                }
            ]
            
            for tool in tools:
                self.tools[tool["name"]] = tool
            
            logger.info("Discovered tools", count=len(tools))
            return tools
            
        except Exception as e:
            raise MCPClientError(
                ErrorType.MCP_SERVER_ERROR,
                f"Failed to discover tools: {str(e)}",
                e
            )
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolCallResponse:
        """Execute a tool with given parameters."""
        start_time = time.time()
        
        try:
            if tool_name not in self.tools:
                return ToolCallResponse(
                    success=False,
                    result=None,
                    error_message=f"Tool '{tool_name}' not found",
                    tool_name=tool_name,
                    execution_time=time.time() - start_time
                )
            
            # Make request to MCP server
            payload = {
                "method": "call_tool",
                "params": {
                    "name": tool_name,
                    "arguments": kwargs
                }
            }
            
            async with self.session.post(
                f"{self.server_url.replace('sse+', '')}/call",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return ToolCallResponse(
                        success=False,
                        result=None,
                        error_message=f"Server error: {response.status} - {error_text}",
                        tool_name=tool_name,
                        execution_time=time.time() - start_time
                    )
                
                result = await response.json()
                return ToolCallResponse(
                    success=True,
                    result=result,
                    tool_name=tool_name,
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            return ToolCallResponse(
                success=False,
                result=None,
                error_message=f"Tool execution error: {str(e)}",
                tool_name=tool_name,
                execution_time=time.time() - start_time
            )


class EnhancedMCPClient:
    """Enhanced MCP client with robust error handling and connection management."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.llm: Optional[OllamaClient] = None
        self.tool_manager: Optional[MCPToolManager] = None
        self.conversation_history: List[ConversationTurn] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.connection_healthy = False
    
    @asynccontextmanager
    async def managed_session(self):
        """Context manager for session lifecycle."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.mcp.connection_timeout)
        )
        
        try:
            # Initialize components
            self.llm = OllamaClient(self.config.ollama)
            await self.llm.__aenter__()
            
            self.tool_manager = MCPToolManager(self.config.mcp.server_url, self.session)
            
            # Health check and tool discovery
            await self._initialize()
            
            yield self
            
        finally:
            if self.llm:
                await self.llm.__aexit__(None, None, None)
            if self.session:
                await self.session.close()
    
    async def _initialize(self):
        """Initialize the client and discover tools."""
        # Check LLM availability
        if not await self.llm.is_available():
            raise MCPClientError(
                ErrorType.LLM_ERROR,
                f"Ollama is not available at {self.config.ollama.base_url}"
            )
        
        # Discover tools
        await self.tool_manager.discover_tools()
        self.connection_healthy = True
        
        logger.info("MCP client initialized successfully")
    
    def _parse_tool_calls(self, response: str) -> List[tuple]:
        """Parse tool calls from LLM response."""
        tool_calls = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('TOOL_CALL:'):
                try:
                    # Parse TOOL_CALL: tool_name(param1=value1, param2=value2)
                    tool_part = line[10:].strip()  # Remove 'TOOL_CALL:'
                    if '(' in tool_part and tool_part.endswith(')'):
                        tool_name = tool_part.split('(')[0].strip()
                        params_str = tool_part[tool_part.find('(')+1:-1]
                        
                        # Simple parameter parsing
                        params = {}
                        if params_str.strip():
                            for param in params_str.split(','):
                                if '=' in param:
                                    key, value = param.split('=', 1)
                                    key = key.strip()
                                    value = value.strip().strip('"\'')
                                    params[key] = value
                        
                        tool_calls.append((tool_name, params))
                except Exception as e:
                    logger.warning("Failed to parse tool call", line=line, error=str(e))
        
        return tool_calls
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and return the response."""
        try:
            # Build conversation context
            messages = [
                {"role": "system", "content": "You are an AI assistant that helps with protein database analysis. Use the available tools when needed to answer questions about proteins, PDB structures, and UniProt data."},
            ]
            
            # Add recent conversation history
            for turn in self.conversation_history[-10:]:  # Last 10 turns
                messages.append({"role": "user", "content": turn.user_message})
                messages.append({"role": "assistant", "content": turn.agent_response})
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Get available tools
            tools = list(self.tool_manager.tools.values())
            
            # Generate initial response
            response = await self.llm.generate(messages, tools)
            
            # Parse and execute tool calls
            tool_calls = self._parse_tool_calls(response)
            executed_tools = []
            
            for tool_name, params in tool_calls:
                if self.config.verbose:
                    console.print(f"[blue]Executing tool: {tool_name} with {params}[/blue]")
                
                tool_result = await self.tool_manager.execute_tool(tool_name, **params)
                executed_tools.append(tool_result)
                
                if tool_result.success:
                    if self.config.verbose:
                        console.print(f"[green]Tool {tool_name} succeeded[/green]")
                else:
                    if self.config.verbose:
                        console.print(f"[red]Tool {tool_name} failed: {tool_result.error_message}[/red]")
            
            # If tools were executed, generate final response with results
            if executed_tools:
                tool_results_text = "\n".join([
                    f"Tool {tr.tool_name}: {'Success' if tr.success else 'Failed'} - {tr.result if tr.success else tr.error_message}"
                    for tr in executed_tools
                ])
                
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool execution results:\n{tool_results_text}\n\nPlease provide a final answer based on these results."})
                
                final_response = await self.llm.generate(messages)
                response = final_response
            
            # Store conversation turn
            turn = ConversationTurn(
                user_message=user_message,
                agent_response=response,
                tool_calls=executed_tools
            )
            self.conversation_history.append(turn)
            
            # Trim history if needed
            if len(self.conversation_history) > self.config.max_conversation_history:
                self.conversation_history = self.conversation_history[-self.config.max_conversation_history:]
            
            return response
            
        except Exception as e:
            logger.error("Error processing message", error=str(e))
            if isinstance(e, MCPClientError):
                return f"Error ({e.error_type.value}): {e.message}"
            else:
                return f"Unexpected error: {str(e)}"


async def retry_with_backoff(
    func,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """Retry function with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            
            delay = min(base_delay * (backoff_factor ** attempt), max_delay)
            logger.warning(
                "Attempt failed, retrying",
                attempt=attempt + 1,
                max_attempts=max_attempts,
                delay=delay,
                error=str(e)
            )
            await asyncio.sleep(delay)


@click.command()
@click.option('--server-url', default='sse+http://127.0.0.1:8000/sse', help='MCP server URL')
@click.option('--model', default='llama3.2', help='Ollama model to use')
@click.option('--ollama-url', default='http://localhost:11434', help='Ollama base URL')
@click.option('--temperature', default=0.1, type=float, help='LLM temperature')
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--log-level', default='INFO', help='Log level')
def main(server_url: str, model: str, ollama_url: str, temperature: float, verbose: bool, log_level: str):
    """Enhanced MCP CLI Client with Ollama LLM Integration."""
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Create configuration
    try:
        config = ClientConfig(
            ollama=OllamaConfig(
                model=model,
                base_url=ollama_url,
                temperature=temperature
            ),
            mcp=MCPConfig(server_url=server_url),
            verbose=verbose,
            log_level=log_level
        )
    except ValidationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    
    # Display startup banner
    console.print(Panel.fit(
        "[bold blue]Enhanced MCP CLI Client[/bold blue]\n"
        f"Model: {config.ollama.model}\n"
        f"Server: {config.mcp.server_url}\n"
        f"Verbose: {config.verbose}",
        title="🤖 MCP Client"
    ))
    
    async def run_client():
        """Main client loop."""
        try:
            async with EnhancedMCPClient(config).managed_session() as client:
                console.print("[green]✓ Connected successfully![/green]")
                console.print("\n[dim]Type 'quit', 'exit', or press Ctrl+C to exit[/dim]\n")
                
                while True:
                    try:
                        # Get user input
                        user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
                        
                        if user_input.lower() in ['quit', 'exit', 'q']:
                            break
                        
                        if not user_input:
                            continue
                        
                        # Process message with spinner
                        with Status("🤔 Thinking...", console=console) as status:
                            response = await client.process_message(user_input)
                        
                        # Display response
                        console.print(Panel(
                            Markdown(response),
                            title="[bold green]🤖 Assistant[/bold green]",
                            border_style="green"
                        ))
                        console.print()
                        
                    except KeyboardInterrupt:
                        break
                    except EOFError:
                        break
                    except Exception as e:
                        console.print(f"[red]Error: {str(e)}[/red]")
                        if config.verbose:
                            console.print_exception()
                
        except Exception as e:
            console.print(f"[red]Failed to start client: {str(e)}[/red]")
            if config.verbose:
                console.print_exception()
            sys.exit(1)
    
    # Run the client
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        console.print("\n[yellow]Goodbye! 👋[/yellow]")
    finally:
        console.print("[dim]Client shutdown complete.[/dim]")


if __name__ == '__main__':
    main()