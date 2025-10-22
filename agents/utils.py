"""Utility helpers shared between agent implementations."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_bot_settings() -> Dict[str, Any]:
    """Load the model configuration for the agents.

    The configuration is cached after the first successful read. If the file
    is missing or malformed the function logs a warning and returns an empty
    dictionary so that the callers can fall back to default model names.
    """

    try:
        with open("agents/bots_settings.json", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("⚠️ Konnte bots_settings.json nicht laden: %s", exc)
        return {}


def get_model_name(agent_key: str, default: str = "llama3") -> str:
    """Return the configured model name for the given agent.

    Falls back to *default* when the configuration file does not contain an
    entry for *agent_key*.
    """

    return load_bot_settings().get(agent_key, default)

