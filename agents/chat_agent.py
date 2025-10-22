"""Chat agent that answers user questions based on earlier analysis."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .utils import get_model_name


logger = logging.getLogger(__name__)


def _normalise_update(value: Any) -> str:
    """Return a clean string representation for optional updates."""

    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        value = "\n".join(str(item) for item in value if str(item).strip())

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Answer follow-up questions using the collected diagnostic context."""

    question = state.get("user_question", "").strip()
    if not question:
        logger.info("[Chat Agent] Keine Frage übergeben – LLM-Aufruf übersprungen.")
        return {
            "chat_response": "",
            "chat_history": state.get("chat_history", []),
        }

    model_name = get_model_name("chat_agent", state)
    llm = ChatOllama(model=model_name, base_url="http://localhost:11434", temperature=0)

    prompt = f"""
You are a car diagnostic assistant AI.
Answer the user's question based on the collected analysis below. Keep answers concise, factual and reference the findings explicitly when useful.

If the user message adds NEW factual information (e.g., new symptoms, noises, car details, replaced parts, clarified causes or solutions), capture it so the main diagnosis can be updated. Minor confirmations without new facts should not trigger an update.

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

User message:
{question}

Respond ONLY in valid JSON with the following structure (use null when a field has no update):
{{
  "chat_response": "...",
  "description_append": "..." | null,
  "car_details": "..." | null,
  "affected_behaviors": "..." | null,
  "noises": "..." | null,
  "changed_parts": "..." | null,
  "possible_causes": "..." | null,
  "possible_solutions": "..." | null,
  "regenerate": true | false
}}

"description_append" should contain only the new facts to append to the original description if the user shared additional context. Leave all fields null if no updates are required. Always reply in the same language as the user.
"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:  # pragma: no cover - LLM specific
            logger.warning("⚠️ LLM-Antwort konnte nicht als JSON interpretiert werden: %s", result)
            raise ValueError("Antwort war kein gültiges JSON") from exc

        response = parsed.get("chat_response", "").strip()

        updates: Dict[str, Any] = {}
        updated_fields: set[str] = set()

        description_append = _normalise_update(parsed.get("description_append"))
        if description_append:
            existing_description = state.get("description_text", "").strip()
            if existing_description:
                updates["description_text"] = f"{existing_description}\n\n{description_append}".strip()
            else:
                updates["description_text"] = description_append
            updated_fields.add("description_text")

        for key in (
            "car_details",
            "affected_behaviors",
            "noises",
            "changed_parts",
            "possible_causes",
            "possible_solutions",
        ):
            value = _normalise_update(parsed.get(key))
            if value:
                updates[key] = value
                updated_fields.add(key)

        regenerate_flag = bool(parsed.get("regenerate", False)) or bool(updates)

        chat_entry = {
            "question": question,
            "response": response,
        }
        chat_history = state.get("chat_history", []) + [chat_entry]

        result_state = {
            "chat_response": response,
            "chat_history": chat_history,
            "regenerate": regenerate_flag,
            "locked_fields": sorted(updated_fields),
        }
        result_state.update(updates)
        return result_state

    except Exception as e:  # pylint: disable=broad-except
        logger.error("❌ Fehler im Chat-Agent: %s", e)
        return {
            "chat_response": "",
            "warning": str(e),
        }
