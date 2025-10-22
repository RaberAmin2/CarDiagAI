"""Agent that generates possible solutions from the derived causes."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .utils import get_language_from_state, get_model_name, localize_phrase


logger = logging.getLogger(__name__)


def possible_solution(state: Dict[str, Any]) -> Dict[str, str]:
    llm = ChatOllama(
        model=get_model_name("possible_solution_agent"),
        base_url="http://localhost:11434",
        temperature=0,
    )

    prompt = f"""
    Task: Based on the possible causes provided, generate a structured solution.
    - Provide clear step-by-step instructions for a mechanic to solve the issue.
    - Indicate if the user can safely perform any of the steps themselves (e.g., checking fluid levels, visually inspecting parts).
    - Base all instructions only on the given possible causes. Do not invent new causes.
    - Always respond in the same language as the input.

    Possible Causes:
    {json.dumps(state.get('possible_causes', ''), indent=2, ensure_ascii=False)}

    Response format (no extra explanations):
    <Translate "POSSIBLE_SOLUTIONS" into the input language>:
    <Translate "Mechanic instructions" into the input language>:
    - step 1
    - step 2
    - step 3

    <Translate "User advice" into the input language>:
    - advice 1
    - advice 2

    If no possible causes are given, respond exactly with the translation
    of "POSSIBLE_SOLUTIONS: None" in the input language.
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"possible_solutions": result.strip()}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("❌ Fehler im Possible-Solution-Agent: %s", exc)
        language = get_language_from_state(state)
        return {"possible_solutions": localize_phrase("possible_solutions_none", language), "warning": str(exc)}
