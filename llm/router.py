"""Select a backend before sending conversation data to a model."""
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from agent.orchestrator import ChatRequest, ChatResponse, Orchestrator
from llm.ollama_client import OllamaClient, OllamaUnavailable

RoutingMode = Literal["auto", "local", "cloud"]
Backend = Literal["local", "cloud"]

_TOOL_CUES = re.compile(
    r"\b(?:github|pull requests?|pr(?:s)?|diffs?|knowledge base|kb|internal|past work|"
    r"my notes|our notes|our repo(?:sitory)?|my repo(?:sitory)?|what did (?:i|we)|"
    r"last week|latest|today|current(?:ly)?|recent(?:ly)?|web search|search the web|"
    r"look up|browse|cite sources|verify online)\b",
    re.IGNORECASE,
)
_COMPLEX_CUES = re.compile(
    r"```|\b(?:compare|refactor\w*|debug\w*|root cause|code review|review (?:my|this|the) code|"
    r"security audit|multi[- ]step|step[- ]by[- ]step|trade[- ]offs?|architecture|"
    r"design (?:a|an|the)|implement|write (?:a|an|the) (?:function|script|class|program))\b",
    re.IGNORECASE,
)


class RoutedRequest(ChatRequest):
    mode: RoutingMode = "auto"


class RoutedResponse(ChatResponse):
    mode: RoutingMode
    backend: Backend
    model: str
    route_reason: str
    fallback: bool = False
    latency_ms: float


class CloudUnavailable(RuntimeError):
    pass


def classify(request: RoutedRequest) -> tuple[Backend, str]:
    if request.mode != "auto":
        return request.mode, "explicit_mode"
    prompts = "\n".join([*(turn.prompt for turn in request.history), request.prompt])
    if _TOOL_CUES.search(prompts):
        return "cloud", "tool_or_fresh_data_cue"
    if _COMPLEX_CUES.search(prompts):
        return "cloud", "complexity_cue"
    # Character count is a cheap context-size proxy with no tokenizer dependency.
    if len(prompts) + sum(len(turn.answer) for turn in request.history) > 2000:
        return "cloud", "long_context"
    return "local", "routine_prompt"


@dataclass
class LLMRouter:
    local: OllamaClient
    cloud: Orchestrator | None
    allow_cloud_fallback: bool = False

    async def chat(self, request: RoutedRequest) -> RoutedResponse:
        started = perf_counter()
        backend, reason = classify(request)
        fallback = False
        if backend == "local":
            messages = [{
                "role": "system",
                "content": (
                    "You are DevBrain, a local developer assistant. Answer using your knowledge "
                    "and the provided conversation. You have no access to external tools. "
                    "State when you lack the information to answer."
                ),
            }]
            for turn in request.history:
                messages.extend([
                    {"role": "user", "content": turn.prompt},
                    {"role": "assistant", "content": turn.answer},
                ])
            messages.append({"role": "user", "content": request.prompt})
            try:
                result = ChatResponse(answer=await self.local.chat(messages), trace=[])
            except OllamaUnavailable:
                if request.mode != "auto" or not self.allow_cloud_fallback or self.cloud is None:
                    raise
                backend, fallback = "cloud", True

        if backend == "cloud":
            if self.cloud is None:
                raise CloudUnavailable("Cloud chat requires OPENAI_API_KEY and DEVBRAIN_AGENT_MODEL.")
            result = await self.cloud.chat(request)
            model = self.cloud.model
        else:
            model = self.local.model

        return RoutedResponse(
            **result.model_dump(), mode=request.mode, backend=backend, model=model,
            route_reason=reason, fallback=fallback, latency_ms=round((perf_counter() - started) * 1000, 2),
        )
