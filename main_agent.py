"""FastAPI entry point for DevBrain's agent chat."""
import asyncio
from contextlib import asynccontextmanager
import logging
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from openai import AsyncOpenAI, OpenAIError
from pydantic import SecretStr, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.orchestrator import AgentProtocolError, ChatRequest, ChatResponse, Orchestrator
from agent.tools_bridge import ToolBridge

logger = logging.getLogger(__name__)


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: SecretStr
    devbrain_agent_model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    tavily_api_key: SecretStr | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AgentSettings()
    api_key = settings.openai_api_key.get_secret_value().strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for agent chat.")
    tavily_key = settings.tavily_api_key
    async with (
        AsyncOpenAI(api_key=api_key, timeout=30, max_retries=1) as client,
        httpx.AsyncClient(timeout=15) as http,
    ):
        app.state.orchestrator = Orchestrator(
            client, settings.devbrain_agent_model,
            ToolBridge(http, tavily_key.get_secret_value().strip() if tavily_key else None),
        )
        yield


app = FastAPI(title="DevBrain Agent", lifespan=lifespan)


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)],
) -> ChatResponse:
    try:
        async with asyncio.timeout(180):
            return await orchestrator.chat(request)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Agent request timed out.") from None
    except (OpenAIError, AgentProtocolError) as exc:
        logger.warning("Agent model request failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Agent model request failed. Check model access and configuration.") from None
