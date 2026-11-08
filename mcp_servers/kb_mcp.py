"""Search the Phase 1 vector store over MCP stdio."""
import asyncio
from contextlib import redirect_stdout
import sys
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field, ValidationError

from embeddings import get_embedder
from retrieval import EmbeddingConfigurationError, QueryRequest, QueryResponse, search_chunks

mcp = FastMCP("DevBrain Knowledge Base")


@mcp.tool()
async def search_devbrain_kb(
    query: Annotated[str, Field(min_length=1, max_length=8000)],
    top_k: Annotated[int, Field(ge=1, le=50, strict=True)] = 3,
) -> QueryResponse:
    """Search DevBrain for relevant chunks. Smaller cosine distances are closer matches."""
    try:
        request = QueryRequest(query=query, top_k=top_k)
    except ValidationError:
        raise ToolError("Query must contain non-whitespace characters.") from None

    try:
        return await asyncio.to_thread(search_chunks, request, validate_configuration=True)
    except EmbeddingConfigurationError as error:
        raise ToolError(str(error)) from None
    except Exception:
        raise ToolError(
            "Knowledge base search failed. Check PostgreSQL connectivity and embedding provider settings."
        ) from None


if __name__ == "__main__":
    # Load model dependencies on the main thread before serving requests.
    with redirect_stdout(sys.stderr):
        try:
            get_embedder()
        except Exception:
            sys.exit("Could not initialize embeddings. Check EMBEDDING_PROVIDER and model access.")
    mcp.run(transport="stdio")
