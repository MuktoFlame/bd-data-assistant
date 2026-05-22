"""
tools/web_search_tool.py
------------------------
LangChain tool for general web search using Tavily, SerpAPI, or DuckDuckGo.

Priority order:
  1. Tavily  (TAVILY_API_KEY)
  2. SerpAPI (SERPAPI_API_KEY)
  3. DuckDuckGo — free, no key required (fallback)
"""

import os
from typing import Optional, Type

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")


class WebSearchTool(BaseTool):
    name: str = "WebSearchTool"
    description: str = (
        "Use this tool for general knowledge queries, policy information, "
        "cultural context, definitions, current events, and anything not "
        "covered by the Bangladesh databases. "
        "Examples: 'What is the role of DGHS in Bangladesh?', "
        "'Healthcare policy in Bangladesh', "
        "'History of Dhaka University', "
        "'Current GDP of Bangladesh'."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        # ── 1. Tavily ──────────────────────────────────────────────────────────
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            return self._search_tavily(query, tavily_key)

        # ── 2. SerpAPI ─────────────────────────────────────────────────────────
        serp_key = os.getenv("SERPAPI_API_KEY")
        if serp_key:
            return self._search_serpapi(query, serp_key)

        # ── 3. DuckDuckGo (free fallback) ──────────────────────────────────────
        return self._search_duckduckgo(query)

    # ── Backends ───────────────────────────────────────────────────────────────

    def _search_tavily(self, query: str, api_key: str) -> str:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, max_results=5)
            results = response.get("results", [])
            if not results:
                return "No results found via Tavily."
            lines = []
            for r in results:
                lines.append(f"**{r.get('title', 'No title')}**")
                lines.append(r.get("content", "")[:500])
                lines.append(f"Source: {r.get('url', '')}\n")
            return "\n".join(lines)
        except Exception as exc:
            return f"[Tavily Error] {exc}"

    def _search_serpapi(self, query: str, api_key: str) -> str:
        try:
            from serpapi import GoogleSearch
            search = GoogleSearch({"q": query, "api_key": api_key, "num": 5})
            results = search.get_dict().get("organic_results", [])
            if not results:
                return "No results found via SerpAPI."
            lines = []
            for r in results:
                lines.append(f"**{r.get('title', '')}**")
                lines.append(r.get("snippet", ""))
                lines.append(f"Source: {r.get('link', '')}\n")
            return "\n".join(lines)
        except Exception as exc:
            return f"[SerpAPI Error] {exc}"

    def _search_duckduckgo(self, query: str) -> str:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            results = list(DDGS().text(query, max_results=5))
            if not results:
                return "No results found via DuckDuckGo."
            lines = []
            for r in results:
                lines.append(f"**{r.get('title', '')}**")
                lines.append(r.get("body", "")[:500])
                lines.append(f"Source: {r.get('href', '')}\n")
            return "\n".join(lines)
        except Exception as exc:
            return (
                f"[DuckDuckGo Error] {exc}\n"
                "Install duckduckgo-search: pip install duckduckgo-search"
            )
