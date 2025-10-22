"""Agent that suggests possible causes based on the gathered evidence."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .utils import get_language_from_state, get_model_name, localize_phrase


logger = logging.getLogger(__name__)


def possible_cause(state: Dict[str, Any]) -> Dict[str, str]:
    llm = ChatOllama(
        model=get_model_name("possible_cause_agent"),
        base_url="http://localhost:11434",
        temperature=0.5,
    )

    prompt = f"""
    Task: Suggest one or more possible technical causes of the reported problem based strictly on the provided information.
    - Consider car details, affected behaviors, noises, and changed parts.
    - Do not invent information that is not mentioned or clearly inferable.
    - Normalize all parts to standard automotive terms.
    - Do not list replaced parts as possible causes unless they are explicitly still suspected to be faulty.
    - Always respond in the same language as the input.

    Car Info: {json.dumps(state.get('car_details', ''), indent=2, ensure_ascii=False)}
    Problem Description: {state.get('description_text', '')}
    Affected Behaviors: {json.dumps(state.get('affected_behaviors', ''), indent=2, ensure_ascii=False)}
    Noises: {json.dumps(state.get('noises', ''), indent=2, ensure_ascii=False)}
    Changed Parts: {json.dumps(state.get('changed_parts', ''), indent=2, ensure_ascii=False)}

    Response format (no explanations outside the structure):
    <Translate "POSSIBLE_CAUSES" into the input language>:
    - cause → reason

    If multiple causes are possible, list them in separate lines.
    If no possible causes can be derived, respond exactly with the translation
    of "POSSIBLE_CAUSES: None" in the input language.
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"possible_causes": result}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("❌ Fehler im Possible-Cause-Agent: %s", exc)
        language = get_language_from_state(state)
        return {"possible_causes": localize_phrase("possible_causes_none", language), "warning": str(exc)}


