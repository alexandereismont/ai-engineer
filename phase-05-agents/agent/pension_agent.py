"""
pension_agent.py — LangGraph agent for pension analytics.

Tools:
    search_regulation(query: str) -> str
        Retrieves relevant regulation chunks from ChromaDB.
    query_fund_metrics(fund_id: str, year: int) -> dict
        Queries DuckDB dim_funds mart for fund data.
    find_related_entities(entity: str) -> list[dict]
        Traverses Neo4j for regulatory entity relationships.

Graph topology:
    START → route_query → [rag_retrieval | sql_query | graph_query | human_checkpoint]
         → generate_response → END
"""

from __future__ import annotations

import os
import re
from typing import Annotated, Any, TypedDict

import chromadb
import duckdb
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = "pension_docs"

DUCKDB_PATH = os.getenv(
    "DUCKDB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "pension_warehouse.duckdb"),
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "pension_secret")

LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Keywords that trigger each routing branch
RAG_KEYWORDS = re.compile(
    r"\b(regulation|article|iorp|ftk|requirement|rule|law|directive|prudent|orsa|esg|"
    r"coverage ratio|dekkingsgraad|recovery plan)\b",
    re.IGNORECASE,
)
SQL_KEYWORDS = re.compile(
    r"\b(fund|return|asset|aum|performance|allocation|metric|data|report|"
    r"quarter|year|2\d{3}|portfolio)\b",
    re.IGNORECASE,
)
GRAPH_KEYWORDS = re.compile(
    r"\b(related|relationship|linked|reference|entity|concept|connect|"
    r"who|what references|graph|network)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """State that flows between nodes in the LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: str  # retrieved context from tools
    tool_calls: int  # running count of tool invocations
    requires_human: bool  # flag that triggers human checkpoint


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@tool
def search_regulation(query: str) -> str:
    """Search the pension regulation knowledge base for relevant text chunks.

    Args:
        query: Natural language question or keyword phrase to search for.

    Returns:
        Concatenated regulation excerpts relevant to the query.
    """
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection(CHROMA_COLLECTION)
        results = collection.query(
            query_texts=[query],
            n_results=4,
            include=["documents", "metadatas"],
        )
        chunks: list[str] = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source = meta.get("source", "unknown")
            article = meta.get("article", "")
            header = f"[{source}" + (f" — {article}]" if article else "]")
            chunks.append(f"{header}\n{doc}")
        return "\n\n---\n\n".join(chunks) if chunks else "No relevant regulation text found."
    except Exception as exc:  # noqa: BLE001
        return f"ChromaDB unavailable: {exc}"


@tool
def query_fund_metrics(fund_id: str, year: int) -> dict:
    """Query fund performance and allocation metrics from the DuckDB data warehouse.

    Args:
        fund_id: The unique fund identifier (e.g. "NL_ABP", "NL_PFZW").
        year: The reporting year (e.g. 2023).

    Returns:
        Dictionary with keys: fund_id, fund_name, aum_eur, coverage_ratio,
        return_ytd, equity_pct, fixed_income_pct, alternatives_pct, year.
        Returns an error dict if the fund is not found.
    """
    try:
        con = duckdb.connect(DUCKDB_PATH, read_only=True)
        row = con.execute(
            """
            SELECT
                f.fund_id,
                f.fund_name,
                m.aum_eur,
                m.coverage_ratio,
                m.return_ytd,
                m.equity_pct,
                m.fixed_income_pct,
                m.alternatives_pct,
                m.reporting_year
            FROM dim_funds f
            JOIN mart_fund_metrics m USING (fund_id)
            WHERE f.fund_id = ?
              AND m.reporting_year = ?
            LIMIT 1
            """,
            [fund_id, year],
        ).fetchone()
        con.close()
        if row is None:
            return {"error": f"No data found for fund '{fund_id}' in year {year}."}
        keys = [
            "fund_id", "fund_name", "aum_eur", "coverage_ratio",
            "return_ytd", "equity_pct", "fixed_income_pct",
            "alternatives_pct", "year",
        ]
        return dict(zip(keys, row))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"DuckDB query failed: {exc}"}


@tool
def find_related_entities(entity: str) -> list[dict]:
    """Traverse the Neo4j knowledge graph to find regulatory entities related to the input.

    Args:
        entity: Name of a concept, article, regulation, or pension fund.

    Returns:
        List of dicts, each with keys: name, type, relationship, direction.
    """
    cypher = """
    MATCH (n)-[r]-(m)
    WHERE toLower(n.name) CONTAINS toLower($entity)
       OR toLower(n.id)   CONTAINS toLower($entity)
    RETURN
        m.name      AS name,
        labels(m)[0] AS type,
        type(r)     AS relationship,
        CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction
    LIMIT 25
    """
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            results = session.run(cypher, entity=entity)
            records = [dict(r) for r in results]
        driver.close()
        return records if records else [{"info": f"No entities related to '{entity}' found."}]
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"Neo4j query failed: {exc}"}]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

SYSTEM_PROMPT = """You are a specialist AI assistant for pension fund analytics and regulation.
You have access to three data sources:
1. A regulation knowledge base (IORP II/III, FTK, Wet toekomst pensioenen)
2. A fund metrics warehouse (AUM, coverage ratios, returns, allocations)
3. A regulatory knowledge graph (entity relationships)

Always ground your answers in retrieved context. If you are unsure, say so clearly
and escalate to a human reviewer. Cite article numbers and regulation names when relevant."""


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def route_query(state: AgentState) -> dict[str, Any]:
    """Determine which retrieval path to take based on the latest user message.

    Checks the most recent HumanMessage against keyword patterns to set the
    routing destination. Also resets the requires_human flag.
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    text = last_human.content if last_human else ""

    # Escalate if explicitly requested
    if re.search(r"\b(human|escalate|unsure|help|confused|expert)\b", text, re.IGNORECASE):
        return {"requires_human": True, "context": state.get("context", "")}

    return {"requires_human": False, "context": state.get("context", "")}


def _route_decision(state: AgentState) -> str:
    """Conditional edge: decide the next node after route_query."""
    if state.get("requires_human"):
        return "human_checkpoint"
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    text = last_human.content if last_human else ""
    if GRAPH_KEYWORDS.search(text):
        return "graph_query"
    if SQL_KEYWORDS.search(text):
        return "sql_query"
    return "rag_retrieval"  # default


def rag_retrieval(state: AgentState) -> dict[str, Any]:
    """Invoke the ChromaDB regulation search tool and store results in context."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = last_human.content if last_human else ""
    context = search_regulation.invoke({"query": query})
    return {
        "context": context,
        "tool_calls": state.get("tool_calls", 0) + 1,
    }


def sql_query(state: AgentState) -> dict[str, Any]:
    """Parse fund_id and year from the user message and query DuckDB metrics."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    text = last_human.content if last_human else ""

    # Simple extraction: look for a fund id pattern and a 4-digit year
    fund_match = re.search(r"\b(NL_\w+|fund[_\s]\w+)\b", text, re.IGNORECASE)
    year_match = re.search(r"\b(20\d{2})\b", text)
    fund_id = fund_match.group(1).upper() if fund_match else "NL_ABP"
    year = int(year_match.group(1)) if year_match else 2023

    result = query_fund_metrics.invoke({"fund_id": fund_id, "year": year})
    context = str(result)
    return {
        "context": context,
        "tool_calls": state.get("tool_calls", 0) + 1,
    }


def graph_query(state: AgentState) -> dict[str, Any]:
    """Extract a key entity from the message and traverse the Neo4j graph."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    text = last_human.content if last_human else ""

    # Prefer capitalised terms as entity names; fall back to the full query
    cap_match = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b", text)
    entity = cap_match[0] if cap_match else text[:60]

    results = find_related_entities.invoke({"entity": entity})
    context = "\n".join(
        f"- {r.get('name', '?')} ({r.get('type', '?')}) "
        f"[{r.get('relationship', '?')} / {r.get('direction', '?')}]"
        for r in results
    )
    return {
        "context": context or "No graph results.",
        "tool_calls": state.get("tool_calls", 0) + 1,
    }


def human_checkpoint(state: AgentState) -> dict[str, Any]:
    """Pause and request human review.

    In LangGraph's interrupt model this node signals the graph to pause;
    the host application is expected to call graph.update_state() with the
    human's response before resuming.
    """
    escalation_msg = AIMessage(
        content=(
            "I am not confident enough to answer this question automatically. "
            "I have escalated to a human reviewer. Please wait for their response."
        )
    )
    return {
        "messages": [escalation_msg],
        "context": "HUMAN_REVIEW_REQUESTED",
        "requires_human": True,
    }


def generate_response(state: AgentState) -> dict[str, Any]:
    """Generate the final answer using the LLM with retrieved context."""
    context = state.get("context", "")

    # Build message list for the LLM
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    if context and context != "HUMAN_REVIEW_REQUESTED":
        messages.append(
            SystemMessage(content=f"Retrieved context:\n\n{context}")
        )
    messages.extend(state["messages"])

    response = llm.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------


def build_graph(checkpointer: MemorySaver | None = None) -> StateGraph:
    """Construct and compile the LangGraph StateGraph.

    Args:
        checkpointer: Optional MemorySaver for conversation persistence.

    Returns:
        Compiled LangGraph application.
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("route_query", route_query)
    builder.add_node("rag_retrieval", rag_retrieval)
    builder.add_node("sql_query", sql_query)
    builder.add_node("graph_query", graph_query)
    builder.add_node("human_checkpoint", human_checkpoint)
    builder.add_node("generate_response", generate_response)

    # Entry point
    builder.add_edge(START, "route_query")

    # Conditional routing after route_query
    builder.add_conditional_edges(
        "route_query",
        _route_decision,
        {
            "rag_retrieval": "rag_retrieval",
            "sql_query": "sql_query",
            "graph_query": "graph_query",
            "human_checkpoint": "human_checkpoint",
        },
    )

    # All retrieval nodes flow into generate_response
    builder.add_edge("rag_retrieval", "generate_response")
    builder.add_edge("sql_query", "generate_response")
    builder.add_edge("graph_query", "generate_response")

    # human_checkpoint ends the graph (caller must resume)
    builder.add_edge("human_checkpoint", END)
    builder.add_edge("generate_response", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_checkpoint"],
    )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def create_agent(persist_memory: bool = True):
    """Create a compiled pension agent.

    Args:
        persist_memory: If True, attach a MemorySaver for multi-turn conversations.

    Returns:
        Compiled LangGraph application ready to invoke.
    """
    checkpointer = MemorySaver() if persist_memory else None
    return build_graph(checkpointer=checkpointer)


def run_agent(query: str, thread_id: str = "default") -> str:
    """Run the agent with a single query and return the final text response.

    Args:
        query: User question in natural language.
        thread_id: Conversation thread identifier for memory isolation.

    Returns:
        Agent's final response string.
    """
    agent = create_agent()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "context": "",
        "tool_calls": 0,
        "requires_human": False,
    }
    final_state = agent.invoke(initial_state, config=config)
    last_ai = next(
        (m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    return last_ai.content if last_ai else "No response generated."


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What does IORP III say about the prudent person principle?"
    )
    print(f"\nQuery: {query}\n")
    print("Response:", run_agent(query))
