"""Agent that extracts noise information from the description."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .utils import get_language_from_state, get_model_name, localize_phrase


logger = logging.getLogger(__name__)


def noise(state: Dict[str, Any]) -> Dict[str, str]:
    llm = ChatOllama(
        model=get_model_name("noise_agent"),
        base_url="http://localhost:11434",
        temperature=0,
    )

    prompt = f"""
    Task: Extract only noise- or sound-related information from the following user description.
    Ignore all unrelated information. If no noise is described, respond with the translation of "NOISES: None".
    Always respond in the same language as the description. Do not translate into another language.

    Description:
    {json.dumps(state.get('description_text', ''), indent=2, ensure_ascii=False)}

    Response format (no explanations, no extra text):
    <Translate "NOISES" into the input language>:
    1. <Translate "Sound" into the input language>: ...
       <Translate "Pattern" into the input language>: ...
       <Translate "Frequency" into the input language>: ...
       <Translate "Details" into the input language>: ...

    If multiple noises are described, list them as 1, 2, 3, ...
    If no noises are described, respond exactly with the translated
    equivalent of "NOISES: None" in the input language.
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"noises": result}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("❌ Fehler im Noise-Agent: %s", exc)
        language = get_language_from_state(state)
        return {"noises": localize_phrase("noise_none", language), "warning": str(exc)}

