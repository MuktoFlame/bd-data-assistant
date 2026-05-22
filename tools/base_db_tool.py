"""
tools/base_db_tool.py
---------------------
Shared base class for all three SQLite-backed LangChain tools.
"""

import os
import sqlite3
import textwrap
from abc import abstractmethod
from typing import Optional, Type

from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field


class DBQueryInput(BaseModel):
    query: str = Field(description="Natural-language question to answer from the database.")


class BaseDBTool(BaseTool):
    """Abstract base class that converts a natural-language query to SQL and runs it."""

    args_schema: Type[BaseModel] = DBQueryInput

    # Sub-classes MUST override these three attributes
    db_path: str = ""
    table_name: str = ""
    table_description: str = ""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if not os.path.exists(self.db_path):
            return (
                f"[Error] Database file not found at '{self.db_path}'. "
                "Please run setup_databases.py first."
            )

        schema = self._get_schema()
        sql = self._generate_sql(query, schema)

        if sql.startswith("[Error]"):
            return sql

        return self._execute_sql(sql, query)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_schema(self) -> str:
        """Return CREATE TABLE statement for context."""
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (self.table_name,),
            )
            row = cur.fetchone()
            return row[0] if row else f"Table '{self.table_name}' not found."

    def _generate_sql(self, natural_query: str, schema: str) -> str:
        """
        Use the LLM (via a simple prompt embedded in the tool) to produce SQL.
        Falls back to a keyword-based heuristic for robustness.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            prompt = textwrap.dedent(f"""
                You are an expert SQLite query writer.
                Table schema:
                {schema}

                Write a single valid SQLite SELECT query to answer this question:
                "{natural_query}"

                Rules:
                - Output ONLY the raw SQL query, no explanation, no markdown.
                - Use LIKE for partial text matching (case-insensitive with LOWER()).
                - Limit results to 20 rows unless a specific count is requested.
                - Never use DROP, DELETE, UPDATE, INSERT, or ALTER.
            """).strip()

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )

            response = llm.invoke(prompt)
            # Gemini may return content as a list of blocks
            content = response.content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = "\n".join(text_parts)
            sql = content.strip().strip("```sql").strip("```").strip()
            return sql

        except Exception as exc:
            return f"[Error] Could not generate SQL: {exc}"

    def _execute_sql(self, sql: str, original_query: str) -> str:
        """Execute the SQL and format results as natural language."""
        try:
            with sqlite3.connect(self.db_path) as con:
                con.row_factory = sqlite3.Row
                cur = con.cursor()
                cur.execute(sql)
                rows = cur.fetchall()

            if not rows:
                return (
                    f"No results found in the {self.table_name} database "
                    f"for your query: '{original_query}'."
                )

            # Format as a readable table-ish string
            col_names = rows[0].keys()
            header = " | ".join(col_names)
            separator = "-" * len(header)
            lines = [header, separator]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))

            result_text = "\n".join(lines)
            return (
                f"Query executed on **{self.table_name}** database.\n"
                f"Found {len(rows)} result(s):\n\n{result_text}"
            )

        except sqlite3.Error as exc:
            return f"[SQL Error] {exc}\nSQL attempted:\n{sql}"
