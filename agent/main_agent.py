"""
Multi-Tool AI Agent for Bangladesh
Main Agent Logic using LangChain create_agent (v1.3+)
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.institutions_tool import InstitutionsDBTool
from tools.hospitals_tool import HospitalsDBTool
from tools.restaurants_tool import RestaurantsDBTool
from tools.web_search_tool import WebSearchTool


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent AI assistant specializing in Bangladesh.
You have access to four tools. You MUST use the appropriate tool for every query.
NEVER answer from your own knowledge when a tool can provide the answer.

## MANDATORY TOOL ROUTING (follow this strictly):

### Database Tools (ALWAYS try these FIRST for data questions):

1. **InstitutionsDBTool** - MUST be used for ANY question about schools, universities,
   colleges, madrasas, polytechnics, or educational/governmental institutions in Bangladesh.
   The database has 34,901 records with columns: name, institute_type, division, location
   (district), thana, education_level, management_type, student_type, affiliation, mpo_status.
   Examples: "How many schools in Dhaka?", "List institutions in Sylhet"

2. **HospitalsDBTool** - MUST be used for ANY question about hospitals, clinics, health
   facilities, or medical centers in Bangladesh.
   The database has 38,886 records with columns: name, hospital_type, division, location
   (district), agency, city_corporation, upazila, private (0=public, 1=private).
   Examples: "How many hospitals in Dhaka?", "List private hospitals in Chittagong"

3. **RestaurantsDBTool** - MUST be used for ANY question about restaurants, food places,
   or eateries in Bangladesh.
   The database has 12,703 records with columns: name, rating, number_of_reviews,
   affluence, address.
   Examples: "Top-rated restaurants?", "Restaurants with rating above 4?"

### Web Search Tool (use ONLY when databases cannot answer):

4. **WebSearchTool** - Use ONLY for general knowledge, policies, definitions, history,
   culture, current events, or when a database tool returns no results.
   Examples: "What is the healthcare policy?", "Role of DGHS in Bangladesh?"

## CRITICAL RULES:
- If the question mentions hospitals, clinics, medical -> USE HospitalsDBTool FIRST
- If the question mentions schools, universities, colleges, institutions -> USE InstitutionsDBTool FIRST
- If the question mentions restaurants, food, dining -> USE RestaurantsDBTool FIRST
- ONLY use WebSearchTool if the DB tool returns no results OR the question is about
  general knowledge/policy/history that is NOT in any database.
- NEVER answer a data question from your own knowledge. ALWAYS query the database first.
- When showing results, say "According to our database..." to indicate the data source.
"""


def build_agent():
    """Create and return the Multi-Tool AI Agent graph."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Uses Google Gemini via free AI Studio API key.
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY in your .env file before running the agent.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=google_key,
    )

    # ── Tools ─────────────────────────────────────────────────────────────────
    tools = [
        InstitutionsDBTool(),
        HospitalsDBTool(),
        RestaurantsDBTool(),
        WebSearchTool(),
    ]

    # ── Agent (LangChain v1.3+ create_agent - LangGraph based) ───────────────
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def run_interactive():
    """Run the agent in an interactive CLI loop."""
    print("\n" + "=" * 60)
    print("  Multi-Tool AI Agent for Bangladesh")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.\n")

    agent = build_agent()
    chat_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            # Build messages list with history + new user message
            messages = list(chat_history) + [HumanMessage(content=user_input)]

            result = agent.invoke({"messages": messages})

            # Extract the final AI response
            output_messages = result.get("messages", [])
            answer = ""
            for msg in reversed(output_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    # Gemini may return content as a list of blocks
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block["text"])
                            elif isinstance(block, str):
                                text_parts.append(block)
                        answer = "\n".join(text_parts)
                    else:
                        answer = str(content)
                    break

            if not answer:
                answer = "I wasn't able to generate a response. Please try rephrasing."

            print(f"\nAgent: {answer}\n")

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        except Exception as exc:
            print(f"\n[Error] {exc}\n")


if __name__ == "__main__":
    run_interactive()
