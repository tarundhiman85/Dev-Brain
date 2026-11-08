"""Expose Phase 2 capabilities and Tavily to the agent."""
from dataclasses import dataclass
import asyncio
from typing import Annotated, Any

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from openai.types.chat import ChatCompletionToolParam
from pydantic import Field, ValidationError, validate_call
from pydantic_core import to_jsonable_python

from mcp_servers import github_mcp, kb_mcp


@dataclass
class ToolBridge:
    http: httpx.AsyncClient
    tavily_api_key: str | None
    allow_keyless_search: bool = False

    async def web_search(
        self, query: Annotated[str, Field(min_length=1, max_length=400)]
    ) -> dict[str, Any]:
        """Search public web sources for current technical information."""
        if not query.strip():
            raise ToolError("Web search requires a nonblank query.")
        if not self.tavily_api_key:
            if not self.allow_keyless_search:
                raise ToolError("Web search requires TAVILY_API_KEY.")
            for backend in ("auto", "brave,bing,mojeek"):
                try:
                    results = await asyncio.to_thread(
                        DDGS(timeout=8).text, query.strip(), max_results=5, backend=backend,
                    )
                    if results:
                        break
                except DDGSException:
                    continue
            else:
                raise ToolError("Public web search returned no results or is temporarily unavailable.")
            return {"results": [
                {"title": item["title"], "url": item["href"], "content": item["body"]}
                for item in results
            ]}
        response = await self.http.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.tavily_api_key}"},
            json={"query": query.strip(), "max_results": 5, "search_depth": "basic"},
        )
        response.raise_for_status()
        return {"results": [
            {"title": item["title"], "url": item["url"], "content": item["content"]}
            for item in response.json()["results"]
        ]}

    async def definitions(self) -> list[ChatCompletionToolParam]:
        web = FastMCP("DevBrain Web Search")
        web.add_tool(self.web_search)
        return [
            {"type": "function", "function": {
                "name": tool.name, "description": tool.description or "",
                "parameters": tool.inputSchema,
            }}
            for server in (kb_mcp.mcp, github_mcp.mcp, web)
            for tool in await server.list_tools()
        ]

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        handlers = {
            "search_devbrain_kb": kb_mcp.search_devbrain_kb,
            "list_recent_prs": github_mcp.list_recent_prs,
            "fetch_pr_details": github_mcp.fetch_pr_details,
            "web_search": self.web_search,
        }
        if name not in handlers:
            raise ToolError("Unknown tool.")
        try:
            result = await validate_call(handlers[name])(**arguments)
        except ValidationError:
            raise ToolError("Invalid tool arguments. Check the tool schema.") from None
        return to_jsonable_python(result)
