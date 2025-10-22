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
    # Sicheres Abrufen mit Fallback auf 'llama3'
    model_name = get_model_name("chat_agent")

    # LLM initialisieren
    llm = ChatOllama(model=model_name, base_url="http://localhost:11434", temperature=0)

    # Prompt definieren
    prompt = f"""
You are a car diagnostic assistant AI.
Use the following problem description and analysis to answer the user's question briefly and helpfully.

Problem description:
{json.dumps(state.get('description_text', ''), indent=2, ensure_ascii=False)}

Identified possible causes:
{json.dumps(state.get('possible_causes', ''), indent=2, ensure_ascii=False)}

User question:
{state.get('user_question', '')}

Respond ONLY in the following JSON format:
{{ "chat_response": "Your response here" }}
"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        # Versuche, JSON zu parsen
        try:
            parsed = json.loads(result)
            response = parsed.get("chat_response", result)
        except json.JSONDecodeError:
            logger.warning("⚠️ LLM-Antwort konnte nicht als JSON interpretiert werden.")
            response = result  # Fallback: rohe Textantwort

        # Chatverlauf aktualisieren
        chat_entry = {
            "question": state.get('user_question', ''),
            "response": response
        }
        chat_history = state.get('chat_history', []) + [chat_entry]

        return {
            "chat_response": response,
            "chat_history": chat_history
        }

    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Fehler im Chat-Agent: %s", e)
        return {
            "chat_response": "",
            "warning": str(e)
        }
