"""Agent that extracts behavior-related information from a description."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

from .fallbacks import fallback_behaviors
from .utils import get_language_from_state, get_model_name, localize_phrase


logger = logging.getLogger(__name__)


def behavior(state: Dict[str, Any]) -> Dict[str, str]:
    try:
        llm = ChatOllama(
            model=get_model_name("behavior_agent", state),
            base_url="http://localhost:11434",
            temperature=0,
        )

        prompt = f"""
Extract only information about the car's behavior from the following description.
This includes driving dynamics and performance issues (e.g., shaking, vibrations, steering problems, braking issues, acceleration issues, stalling, pulling, loss of power).
Ignore noises, replaced parts, or vehicle specifications.

Normalize the behaviors into clear, concise automotive terms.
If the user uses informal phrases, rewrite them as standard behavior descriptions.

Description:
{state.get('description_text', '')}

Response format (no explanations, no extra text):
<Translate "Affected behaviors" into the input language>:
- behavior1
- behavior2
- behavior3

If no behaviors can be identified, respond exactly with the translation of
"No affected behaviors identified" in the language used by the user.

Always answer in the same language as the input.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

        logger.info("[Behavior Agent] Output: %s", result)

        return {
            "affected_behaviors": result
        }

    except Exception as e:  # pylint: disable=broad-except
        logger.error("[Behavior Agent] Error: %s", e)
        language = get_language_from_state(state)
        fallback = fallback_behaviors(state.get("description_text", ""), language)
        if not fallback:
            fallback = localize_phrase("behavior_none", language)
        return {
            "affected_behaviors": fallback,
            "warning": str(e)
        }

