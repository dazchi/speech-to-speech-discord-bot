"""Google search through Serper.dev (needs SERPER_API_KEY)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

SERPER_URL = "https://google.serper.dev/search"
MAX_RESULTS = 5

TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web (Google) for current or uncertain information: news, weather, prices, exchange "
            "rates, scores, schedules, recent events, or facts you are not sure about. Returns a direct answer "
            "when Google has one, plus the top results with titles and snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short keyword query, e.g. '台北 今天 天氣' or 'USD TWD exchange rate'.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
]


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")
    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"error": "query must be a non-empty string"}
    key = os.environ.get("SERPER_API_KEY", "").strip()
    if not key:
        return {"query": query, "error": "SERPER_API_KEY is not set."}
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post(
                SERPER_URL,
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={"q": query, "num": MAX_RESULTS},
            )
    except httpx.RequestError:
        return {"query": query, "error": "Search provider is temporarily unreachable."}
    if response.status_code != 200:
        return {"query": query, "error": f"Search provider error ({response.status_code})"}
    data = response.json()
    box = data.get("answerBox") or {}
    answer = box.get("answer") or box.get("snippet") or (data.get("knowledgeGraph") or {}).get("description")
    return {
        "query": query,
        "searched_at": datetime.now(timezone.utc).date().isoformat(),
        "answer": answer,
        "results": [
            {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "url": r.get("link", "")}
            for r in (data.get("organic") or [])[:MAX_RESULTS]
        ],
    }
