"""Agent that extracts already replaced parts from the description."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .fallbacks import fallback_changed_parts
from .utils import get_language_from_state, get_model_name, localize_phrase


logger = logging.getLogger(__name__)


def new_parts(state: Dict[str, Any]) -> Dict[str, str]:
    llm = ChatOllama(
        model=get_model_name("new_parts_agent", state),
        base_url="http://localhost:11434",
        temperature=0,
    )

    prompt = f"""
Task: Extract only the parts that the user explicitly mentions as already replaced, exchanged, or newly installed.
Normalize all mentioned parts to their standard automotive part names.
Do not include broken, old, or suggested parts.
Always respond in the same language as the description. Do not translate into another language.

Description:
{json.dumps(state.get('description_text', ''), indent=2, ensure_ascii=False)}

Response format (no explanations, no extra text):
<Translate "NEW_PARTS" into the input language>:
- part1
- part2
- part3

If no replaced parts are mentioned, respond exactly with the translated
equivalent of "NEW_PARTS: None" in the input language.
"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"changed_parts": result}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("❌ Fehler im New-Parts-Agent: %s", exc)
        language = get_language_from_state(state)
        fallback = fallback_changed_parts(state.get("description_text", ""), language)
        if not fallback:
            fallback = localize_phrase("new_parts_none", language)
        return {"changed_parts": fallback, "warning": str(exc)}
