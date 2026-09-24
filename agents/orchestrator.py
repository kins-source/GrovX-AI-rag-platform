import os
from functools import lru_cache
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage
from agents.llm_provider import get_llm
from agents.sql_agent import query_sales_database
from agents.retriever_agent import retrieve_enterprise_documents
from agents.guardrails import is_safe_query, validate_grounding
from rag.cache import query_cache
from utils.logger import logger
from utils.telemetry import track_latency
from dotenv import load_dotenv

load_dotenv()

# Define the tools available to the orchestrator
tools = [query_sales_database, retrieve_enterprise_documents]


def _nvidia_agent_response(user_query, api_key, model):
    """Execute local tools explicitly because NVIDIA NIM tool calls are not reliable here."""
    query_lower = user_query.lower()
    tool_result = None

    if any(keyword in query_lower for keyword in ("revenue", "sales", "orders", "customer count", "transactions")):
        if "average" in query_lower:
            sql_query = "SELECT AVG(total_price) FROM orders;"
        elif "count" in query_lower or "number" in query_lower or "total orders" in query_lower:
            sql_query = "SELECT COUNT(*) FROM orders;"
        else:
            sql_query = "SELECT SUM(total_price) FROM orders;"
        tool_result = query_sales_database.invoke(sql_query)
    elif any(keyword in query_lower for keyword in ("document", "pdf", "policy", "compliance", "uploaded")):
        tool_result = retrieve_enterprise_documents.invoke(user_query)

    system_prompt = (
        "You are an Enterprise AI Knowledge Assistant. Answer the user's question directly and concisely. "
        "Use the supplied database or document context when present. Do not mention internal tools or implementation details."
    )
    if tool_result is not None:
        system_prompt += f"\n\nRetrieved context:\n{tool_result}"

    response = get_llm("nvidia", api_key, model).invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ])
    messages = [response]
    if tool_result is not None:
        messages.insert(0, ToolMessage(content=str(tool_result), tool_call_id="nvidia-direct-tool"))
    return {"messages": messages}


@lru_cache(maxsize=12)
def get_agent(provider=None, api_key=None, model=None):
    return create_react_agent(get_llm(provider, api_key, model), tools)

@track_latency
def process_query(user_query: str, provider=None, api_key=None, model=None) -> dict:
    """
    Main entry point for handling user queries.
    Applies guardrails, checks cache, executes the agent, and formats the response.
    """
    logger.info(f"Processing query: {user_query}")
    
    # 1. Guardrails: Input validation
    safe, reason = is_safe_query(user_query)
    if not safe:
        return {"answer": reason, "sources": []}
    
    # 2. Cache Lookup
    cached_response = query_cache.get(user_query)
    if cached_response:
        return {"answer": cached_response, "sources": ["Cache"]}
    
    # 3. Execute Agent Orchestrator
    try:
        # We append a structured prompt instructing the agent
        system_prompt = (
            "You are an Enterprise AI Knowledge Assistant. "
            "You have access to a Sales SQL Database tool and an Enterprise Document Database tool. "
            "CRITICAL: If a user asks about numbers, revenue, sales, or orders, you MUST USE the query_sales_database tool. "
            "CRITICAL: If a user asks about documents, compliance, or general knowledge, you MUST USE the retrieve_enterprise_documents tool. "
            "DO NOT ask the user for a query string or tool permissions. You MUST independently extract the required parameters from the user's message and EXECUTE THE TOOL IMMEDIATELY. "
            "DO NOT explain how to query the database in natural language. DO NOT return code to the user as your final answer. "
            "YOU MUST EXECUTE THE TOOL FIRST. Then, return ONLY the final computed answer or summary based on the tool result."
        )
        
        # In LangGraph create_react_agent, we pass a list of messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Invoke the LangGraph agent
        if (provider or os.getenv("LLM_PROVIDER", "ollama")).lower() == "nvidia":
            response_state = _nvidia_agent_response(user_query, api_key, model)
        else:
            response_state = get_agent(provider, api_key, model).invoke({"messages": messages})
        
        # Extract the AI's final answer
        final_answer = str(response_state["messages"][-1].content)
        
        # Clean leaked tool JSON traces (e.g. [{"name": "tool_name", ...}]) from local model outputs
        import re
        final_answer = re.sub(r'\[\s*\{.*?"name"\s*:.*?"arguments"\s*:.*?\}\s*\]', '', final_answer, flags=re.DOTALL).strip()
        
        # If the trace was the ONLY output (empty final synthesis), fallback to the actual executed tool result
        if not final_answer:
            for msg in reversed(response_state["messages"]):
                if getattr(msg, "type", "") == "tool":
                    final_answer = str(msg.content).strip()
                    break
        
        if not final_answer:
            final_answer = "Could not synthesize a final answer."
        
        # 4. Guardrails: Output Verification (stubbed)
        if not validate_grounding(final_answer, ""):
            final_answer += "\n\n(Disclaimer: This response could not be fully verified against context.)"
        
        # Cache the successful response
        query_cache.set(user_query, final_answer)
        
        # Extract sources from tool messages if any
        sources = []
        import json
        import re
        for msg in response_state["messages"]:
            if getattr(msg, "type", "") == "tool":
                try:
                    # Attempt to parse JSON structure containing sources list
                    parsed_content = json.loads(str(msg.content))
                    if "sources" in parsed_content:
                        sources.extend(parsed_content["sources"])
                except Exception:
                    # Fallback regex parsing
                    if "Source [" in str(msg.content):
                        matches = re.findall(r"Source \[(.*?)\]", str(msg.content))
                        sources.extend(matches)
                
        # Deduplicate sources and cleanly sort
        sources = sorted(list(set(sources)))
        
        return {"answer": final_answer, "sources": sources}
        
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        if (provider or os.getenv("LLM_PROVIDER", "ollama")).lower() == "nvidia" and "Function" in str(e):
            return {
                "answer": "NVIDIA NIM rejected this request. Rotate NVIDIA_NIM_API_KEY in .env and restart the backend.",
                "sources": [],
            }
        return {"answer": f"Internal system error occurred: {str(e)}", "sources": []}
