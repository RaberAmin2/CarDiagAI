"""Chat agent that answers user questions based on earlier analysis."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .utils import get_model_name


logger = logging.getLogger(__name__)


def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Answer follow-up questions using the collected diagnostic context."""

    question = state.get("user_question", "").strip()
    if not question:
        logger.info("[Chat Agent] Keine Frage übergeben – LLM-Aufruf übersprungen.")
        return {
            "chat_response": "",
            "chat_history": state.get("chat_history", []),
        }

    model_name = get_model_name("chat_agent")
    llm = ChatOllama(model=model_name, base_url="http://localhost:11434", temperature=0)

    prompt = f"""
You are a car diagnostic assistant AI.
Answer the user's question based on the collected analysis below. Keep answers concise, factual and reference the findings explicitly when useful.

Problem description:
{json.dumps(state.get('description_text', ''), indent=2, ensure_ascii=False)}

Car details:
{json.dumps(state.get('car_details', ''), indent=2, ensure_ascii=False)}

Affected behaviors:
{json.dumps(state.get('affected_behaviors', ''), indent=2, ensure_ascii=False)}

Detected noises:
{json.dumps(state.get('noises', ''), indent=2, ensure_ascii=False)}

Changed parts:
{json.dumps(state.get('changed_parts', ''), indent=2, ensure_ascii=False)}

Possible causes:
{json.dumps(state.get('possible_causes', ''), indent=2, ensure_ascii=False)}

Possible solutions:
{json.dumps(state.get('possible_solutions', ''), indent=2, ensure_ascii=False)}

User question:
{question}

Respond ONLY in the following JSON format:
{{ "chat_response": "Your response here" }}
"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        try:
            parsed = json.loads(result)
            response = parsed.get("chat_response", result)
        except json.JSONDecodeError:
            logger.warning("⚠️ LLM-Antwort konnte nicht als JSON interpretiert werden.")
            response = result

        chat_entry = {
            "question": question,
            "response": response,
        }
        chat_history = state.get("chat_history", []) + [chat_entry]

        return {
            "chat_response": response,
            "chat_history": chat_history,
        }

    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Fehler im Chat-Agent: %s", e)
        return {
            "chat_response": "",
            "warning": str(e),
        }
