"""Utility helpers shared between agent implementations."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, Iterable, List

from langdetect import DetectorFactory, LangDetectException, detect


logger = logging.getLogger(__name__)


# Make language detection deterministic across runs
DetectorFactory.seed = 0


_SUPPORTED_LANGUAGES = {
    "en": "en",
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
}


_LOCALIZED_FALLBACKS: Dict[str, Dict[str, str]] = {
    "en": {
        "behavior_none": "No affected behaviors identified",
        "noise_none": "NOISES: None",
        "new_parts_none": "NEW_PARTS: None",
        "possible_causes_none": "POSSIBLE_CAUSES: None",
        "possible_solutions_none": "POSSIBLE_SOLUTIONS: None",
    },
    "de": {
        "behavior_none": "Keine betroffenen Verhaltensweisen identifiziert",
        "noise_none": "GERÄUSCHE: Keine",
        "new_parts_none": "NEUE_TEILE: Keine",
        "possible_causes_none": "MÖGLICHE_URSACHEN: Keine",
        "possible_solutions_none": "LÖSUNGSVORSCHLÄGE: Keine",
    },
    "es": {
        "behavior_none": "No se identificaron comportamientos afectados",
        "noise_none": "RUIDOS: Ninguno",
        "new_parts_none": "PIEZAS_NUEVAS: Ninguna",
        "possible_causes_none": "POSIBLES_CAUSAS: Ninguna",
        "possible_solutions_none": "POSIBLES_SOLUCIONES: Ninguna",
    },
    "fr": {
        "behavior_none": "Aucun comportement affecté identifié",
        "noise_none": "BRUITS : Aucun",
        "new_parts_none": "PIÈCES_REMPLACÉES : Aucune",
        "possible_causes_none": "CAUSES_POSSIBLES : Aucune",
        "possible_solutions_none": "SOLUTIONS_POSSIBLES : Aucune",
    },
    "it": {
        "behavior_none": "Nessun comportamento interessato identificato",
        "noise_none": "RUMORI: Nessuno",
        "new_parts_none": "PARTI_SOSTITUITE: Nessuna",
        "possible_causes_none": "CAUSE_POSSIBILI: Nessuna",
        "possible_solutions_none": "SOLUZIONI_POSSIBILI: Nessuna",
    },
    "pt": {
        "behavior_none": "Nenhum comportamento afetado identificado",
        "noise_none": "RUÍDOS: Nenhum",
        "new_parts_none": "PEÇAS_SUBSTITUÍDAS: Nenhuma",
        "possible_causes_none": "CAUSAS_POSSÍVEIS: Nenhuma",
        "possible_solutions_none": "SOLUÇÕES_POSSÍVEIS: Nenhuma",
    },
    "nl": {
        "behavior_none": "Geen getroffen rijgedrag gevonden",
        "noise_none": "GELUIDEN: Geen",
        "new_parts_none": "NIEUWE_ONDERDELEN: Geen",
        "possible_causes_none": "MOGELIJKE_OORZAKEN: Geen",
        "possible_solutions_none": "MOGELIJKE_OPLOSSINGEN: Geen",
    },
}


def detect_language(text: str, fallback: str = "en") -> str:
    """Best-effort detection of the language used in *text*."""

    if not text or not text.strip():
        return fallback

    candidate = text.strip()

    try:
        detected = detect(candidate)
    except LangDetectException:
        logger.debug("⚠️ Sprachenerkennung fehlgeschlagen, verwende Fallback '%s'", fallback)
        return fallback

    language = detected.split("-")[0]
    return _SUPPORTED_LANGUAGES.get(language, fallback)


def get_language_from_state(state: Dict[str, Any], fallback: str = "en") -> str:
    """Infer the most likely language from the diagnostic *state*."""

    for key in ("user_question", "description_text", "car_details"):
        value = state.get(key, "")
        if value and value.strip():
            return detect_language(value, fallback=fallback)

    return fallback


def localize_phrase(key: str, language: str) -> str:
    """Return a localized fallback phrase for *key* in the requested language."""

    phrases = _LOCALIZED_FALLBACKS.get(language)
    if phrases and key in phrases:
        return phrases[key]

    default_phrases = _LOCALIZED_FALLBACKS.get("en", {})
    return default_phrases.get(key, key)


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


def _gather_relevant_text(agent_key: str, state: Dict[str, Any] | None) -> str:
    """Collect text snippets that describe the agent task context."""

    if not state:
        return ""

    key_map: Dict[str, Iterable[str]] = {
        "identify_car_agent": ("description_text",),
        "behavior_agent": ("description_text",),
        "noise_agent": ("description_text",),
        "new_parts_agent": ("description_text",),
        "possible_cause_agent": (
            "description_text",
            "car_details",
            "affected_behaviors",
            "noises",
            "changed_parts",
        ),
        "possible_solution_agent": (
            "description_text",
            "car_details",
            "affected_behaviors",
            "noises",
            "possible_causes",
            "changed_parts",
        ),
        "chat_agent": ("user_question",),
    }

    relevant_keys = key_map.get(agent_key)
    if not relevant_keys:
        return state.get("description_text", "")

    parts: List[str] = []
    for key in relevant_keys:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    return "\n\n".join(parts)


def _score_text_complexity(text: str) -> int:
    """Return a heuristic complexity score for *text*."""

    if not text or not text.strip():
        return 0

    words = text.split()
    word_count = len(words)
    unique_terms = len({word.lower().strip(",.;:") for word in words if word})
    sentence_markers = sum(text.count(marker) for marker in ".!?")
    list_markers = text.count("\n-") + text.count("\n*")
    paragraph_breaks = text.count("\n\n")

    score = word_count
    score += unique_terms // 2
    score += sentence_markers * 3
    score += list_markers * 6
    score += paragraph_breaks * 4

    return score


def _additional_context_score(agent_key: str, state: Dict[str, Any] | None) -> int:
    if not state:
        return 0

    supporting_keys: Dict[str, Iterable[str]] = {
        "possible_cause_agent": (
            "car_details",
            "affected_behaviors",
            "noises",
            "changed_parts",
        ),
        "possible_solution_agent": (
            "car_details",
            "affected_behaviors",
            "noises",
            "possible_causes",
            "changed_parts",
        ),
        "chat_agent": ("chat_history",),
    }

    keys = supporting_keys.get(agent_key, ())
    bonus = 0
    for key in keys:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            bonus += 25
        elif isinstance(value, list) and value:
            bonus += 15 + 5 * len(value)
    return bonus


def determine_task_complexity(agent_key: str, state: Dict[str, Any] | None) -> str:
    """Estimate the complexity tier for the agent call."""

    text = _gather_relevant_text(agent_key, state)
    score = _score_text_complexity(text)
    score += _additional_context_score(agent_key, state)

    if score <= 160:
        complexity = "simple"
    elif score <= 340:
        complexity = "moderate"
    else:
        complexity = "complex"

    logger.debug(
        "[Model Selector] Agent: %s | Score: %s | Complexity: %s",
        agent_key,
        score,
        complexity,
    )

    return complexity


def get_model_name(
    agent_key: str,
    state: Dict[str, Any] | None = None,
    default: str = "llama3",
) -> str:
    """Return the configured model name for the given agent.

    Supports tiered configuration via ``simple``/``moderate``/``complex`` keys.
    """

    settings = load_bot_settings()
    config = settings.get(agent_key)
    defaults = settings.get("defaults", {})

    if isinstance(config, dict):
        complexity = determine_task_complexity(agent_key, state)
        return (
            config.get(complexity)
            or config.get("default")
            or defaults.get(complexity)
            or defaults.get("default")
            or default
        )

    if isinstance(config, str):
        return config

    if state is not None:
        complexity = determine_task_complexity(agent_key, state)
        return (
            defaults.get(complexity)
            or defaults.get("default")
            or default
        )

    return defaults.get("default", default)

