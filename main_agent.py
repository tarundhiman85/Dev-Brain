"""FastAPI entry point for DevBrain's agent chat."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import logging
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from openai import AsyncOpenAI, OpenAIError
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.orchestrator import AgentProtocolError, Orchestrator
from agent.tools_bridge import ToolBridge
from config import get_settings
from llm.ollama_client import OllamaClient, OllamaUnavailable
from llm.router import CloudUnavailable, LLMRouter, RoutedRequest, RoutedResponse

logger = logging.getLogger(__name__)


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: SecretStr | None = None
    devbrain_agent_model: str = ""
    tavily_api_key: SecretStr | None = None
    ollama_chat_model: str = "llama3.2"
    devbrain_allow_cloud_fallback: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = AgentSettings()
    async with AsyncExitStack() as stack:
        local_http = await stack.enter_async_context(httpx.AsyncClient(
            base_url=str(get_settings().ollama_base_url),
            timeout=httpx.Timeout(60, connect=3),
            trust_env=False,
        ))
        cloud = None
        api_key = settings.openai_api_key
        model = settings.devbrain_agent_model.strip()
        if api_key and api_key.get_secret_value().strip() and model:
            client = await stack.enter_async_context(
                AsyncOpenAI(api_key=api_key.get_secret_value().strip(), timeout=30, max_retries=1)
            )
            http = await stack.enter_async_context(httpx.AsyncClient(timeout=15))
            tavily_key = settings.tavily_api_key
            cloud = Orchestrator(client, model, ToolBridge(
                http, tavily_key.get_secret_value().strip() if tavily_key else None,
            ))
        app.state.router = LLMRouter(
            OllamaClient(local_http, settings.ollama_chat_model), cloud,
            settings.devbrain_allow_cloud_fallback,
        )
        yield


app = FastAPI(title="DevBrain Agent", lifespan=lifespan)


def get_router(request: Request) -> LLMRouter:
    return request.app.state.router


@app.post("/api/v1/agent/chat", response_model=RoutedResponse)
async def chat(
    request: RoutedRequest,
    router: Annotated[LLMRouter, Depends(get_router)],
) -> RoutedResponse:
    try:
        async with asyncio.timeout(180):
            return await router.chat(request)
    except (OllamaUnavailable, CloudUnavailable) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Agent request timed out.") from None
    except (OpenAIError, AgentProtocolError) as exc:
        logger.warning("Agent model request failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Agent model request failed. Check model access and configuration.") from None
