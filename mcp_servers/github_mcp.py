"""Read-only GitHub tools served over MCP stdio."""
import asyncio
from functools import lru_cache
from typing import Annotated, Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

mcp = FastMCP("DevBrain GitHub")
RepoName = Annotated[str, Field(pattern=r"^[A-Za-z0-9-]+/[A-Za-z0-9_.-]+$", max_length=140)]


class GitHubSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    github_token: SecretStr | None = None


@lru_cache
def settings() -> GitHubSettings:
    return GitHubSettings()


def github_client(repo_name: str) -> httpx.AsyncClient:
    if repo_name.split("/")[-1] in {".", ".."}:
        raise ToolError("Provide a repository as owner/repo.")
    token = settings().github_token
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token and token.get_secret_value():
        headers["Authorization"] = f"Bearer {token.get_secret_value()}"
    return httpx.AsyncClient(base_url=f"https://api.github.com/repos/{repo_name}/",
                             headers=headers, timeout=15)


async def get(client: httpx.AsyncClient, path: str, **params: str | int) -> httpx.Response:
    try:
        response = await client.get(path, params=params)
    except httpx.RequestError:
        raise ToolError("Could not reach GitHub. Check your network and try again.") from None
    if response.status_code == 404:
        raise ToolError("GitHub could not find the repository or PR. Check the name and token access.")
    if response.status_code in (401, 403, 429):
        raise ToolError(f"GitHub denied the request (HTTP {response.status_code}). Check token access and rate limits.")
    if not response.is_success:
        raise ToolError(f"GitHub returned HTTP {response.status_code}.")
    return response


def summary(pr: dict[str, Any]) -> dict[str, Any]:
    return {key: pr.get(key) for key in
            ("number", "title", "state", "html_url", "created_at", "updated_at", "draft")}


async def first_page(client: httpx.AsyncClient, path: str) -> tuple[list[dict[str, Any]], bool]:
    response = await get(client, path, per_page=100)
    return response.json(), "next" in response.links


@mcp.tool()
async def list_recent_prs(repo_name: RepoName,
                          limit: Annotated[int, Field(ge=1, le=100, strict=True)] = 5) -> dict[str, Any]:
    """List PRs in owner/repo, newest created first, including closed and merged PRs."""
    async with github_client(repo_name) as client:
        response = await get(client, "pulls", state="all", sort="created", direction="desc", per_page=limit)
    return {"pull_requests": [summary(pr) for pr in response.json()]}


@mcp.tool()
async def fetch_pr_details(repo_name: RepoName,
                           pr_number: Annotated[int, Field(ge=1, strict=True)]) -> dict[str, Any]:
    """Fetch a PR body, conversation and inline review comments, and per-file diff patches."""
    async with github_client(repo_name) as client:
        pr = (await get(client, f"pulls/{pr_number}")).json()
        discussion, review, changes = await asyncio.gather(
            first_page(client, f"issues/{pr_number}/comments"),
            first_page(client, f"pulls/{pr_number}/comments"),
            first_page(client, f"pulls/{pr_number}/files"),
        )
    comments, more_comments = discussion
    review_comments, more_reviews = review
    files, more_files = changes
    return {
        **summary(pr),
        "body": pr.get("body"),
        "merged": pr.get("merged"),
        "comments": [{key: item.get(key) for key in ("body", "html_url", "created_at")}
                     for item in comments],
        "review_comments": [{key: item.get(key) for key in ("body", "path", "line", "html_url")}
                            for item in review_comments],
        "files": [{key: item.get(key) for key in
                   ("filename", "previous_filename", "status", "additions", "deletions", "patch")}
                  for item in files],
        "has_more": {"comments": more_comments, "review_comments": more_reviews, "files": more_files},
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
