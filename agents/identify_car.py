"""Agent to extract structured vehicle information from a description."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .utils import get_model_name


logger = logging.getLogger(__name__)


def identify_car(state: Dict[str, Any]) -> Dict[str, str]:
    """Extract normalized car details from the problem description."""

    description = state.get("description_text", "")
    model_name = get_model_name("identify_car_agent", state)

    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0,
    )

    prompt = f"""
Task: Extract vehicle details from the following problem description.
- Normalize model names and technical details (e.g., Golf VII → Golf 7).
- If a detail can be reasonably inferred from the description (e.g., "Golf VII" → Brand: VW, Model: Golf 7), include it.
- If a detail cannot be identified or inferred with certainty, write "Unknown".
- Always respond in the same language as the description. Do not translate.

Description:
{json.dumps(description, indent=2, ensure_ascii=False)}

Response format (no extra words, no explanations):
<Translate "CAR_DETAILS" into the input language>:
- <Translate "Brand" into the input language>: ...
- <Translate "Model" into the input language>: ...
- <Translate "Engine" into the input language>: ...
- <Translate "Transmission" into the input language>: ...
- <Translate "Year" into the input language>: ...
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        return {"car_details": result}
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("❌ Fehler im Identify-Car-Agent: %s", exc)
        return {"car_details": "", "warning": str(exc)}

