"""Fallback generators that provide meaningful output when the LLM is unavailable."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

_LANG_STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "car_details_header": "CAR_DETAILS",
        "brand_label": "Brand",
        "model_label": "Model",
        "engine_label": "Engine",
        "transmission_label": "Transmission",
        "year_label": "Year",
        "unknown_value": "Unknown",
        "behaviors_header": "Affected behaviors",
        "changed_parts_header": "NEW_PARTS",
        "possible_causes_header": "POSSIBLE_CAUSES",
        "possible_solutions_header": "POSSIBLE_SOLUTIONS",
        "mechanic_instructions_header": "Mechanic instructions",
        "user_advice_header": "User advice",
    },
    "de": {
        "car_details_header": "FAHRZEUGDETAILS",
        "brand_label": "Marke",
        "model_label": "Modell",
        "engine_label": "Motor",
        "transmission_label": "Getriebe",
        "year_label": "Baujahr",
        "unknown_value": "Unbekannt",
        "behaviors_header": "Betroffene Verhaltensweisen",
        "changed_parts_header": "ERSETZTE_TEILE",
        "possible_causes_header": "MÖGLICHE_URSACHEN",
        "possible_solutions_header": "LÖSUNGSVORSCHLÄGE",
        "mechanic_instructions_header": "Anleitung für die Werkstatt",
        "user_advice_header": "Hinweise für den Fahrer",
    },
}


_REPLACED_TRIGGERS: Dict[str, Iterable[str]] = {
    "en": ("replaced", "replace", "changed", "new", "installed"),
    "de": ("ersetzt", "ausgetauscht", "gewechselt", "erneuert", "neu"),
}


def _normalise_language(language: str) -> str:
    if language in _LANG_STRINGS:
        return language
    return "en"


def _strings(language: str) -> Dict[str, str]:
    lang = _normalise_language(language)
    return _LANG_STRINGS[lang]


def _normalise_text(text: str) -> str:
    text = text.lower()
    text = text.replace("ß", "ss")
    return text


_ISSUE_PROFILES: List[Dict[str, object]] = [
    {
        "id": "vibration",
        "keywords": {
            "en": ("vibration", "vibrations", "shaking", "shudder", "wobble"),
            "de": (
                "vibration",
                "vibrationen",
                "ruckeln",
                "schuettel",
                "schuttel",
                "zittern",
                "wackeln",
            ),
        },
        "behaviors": {
            "en": ("- Persistent vibrations while driving",),
            "de": ("- Anhaltende Vibrationen während der Fahrt",),
        },
        "causes": {
            "en": (
                "- Wheel imbalance or worn suspension components → vibrations at all speeds",
            ),
            "de": (
                "- Unwucht der Räder oder verschlissene Fahrwerkskomponenten → Vibrationen bei allen Geschwindigkeiten",
            ),
        },
        "mechanic_steps": {
            "en": (
                "Check wheel balance and tire condition",
                "Inspect tie rods, control arms and suspension bushings",
                "Perform an alignment after repairs",
            ),
            "de": (
                "Radauswuchtung und Reifen prüfen",
                "Spurstangen, Querlenker und Fahrwerkslager kontrollieren",
                "Nach Reparaturen eine Achsvermessung durchführen",
            ),
        },
        "user_advice": {
            "en": (
                "Avoid high speeds until the vibration is resolved",
                "Monitor the replaced tie rod for play or looseness",
            ),
            "de": (
                "Hohe Geschwindigkeiten vermeiden, bis die Vibration behoben ist",
                "Die ersetzte Spurstange auf Spiel oder Lockerheit beobachten",
            ),
        },
        "replaced_parts": {
            "en": (
                {"keywords": ("tie rod",), "label": "Tie rod (right side)"},
            ),
            "de": (
                {"keywords": ("spurstange",), "label": "Spurstange (rechte Seite)"},
            ),
        },
    }
]


_BRAND_PATTERNS: Dict[str, Dict[str, object]] = {
    "volvo": {"brand": "Volvo", "models": ("c30", "s60", "xc60", "xc90")},
    "bmw": {"brand": "BMW", "models": ("x5", "x3", "3er", "5er", "i3")},
    "audi": {"brand": "Audi", "models": ("a3", "a4", "a6", "q5")},
    "mercedes": {"brand": "Mercedes-Benz", "models": ("c-klasse", "e-klasse", "glc")},
    "vw": {"brand": "Volkswagen", "models": ("golf", "passat", "tiguan")},
    "volkswagen": {"brand": "Volkswagen", "models": ("golf", "passat", "tiguan")},
    "ford": {"brand": "Ford", "models": ("focus", "fiesta", "mustang")},
    "toyota": {"brand": "Toyota", "models": ("corolla", "yaris", "rav4")},
}


def _detect_profiles(description: str, language: str) -> List[Dict[str, object]]:
    text = _normalise_text(description)
    lang = _normalise_language(language)
    matches: List[Dict[str, object]] = []
    for profile in _ISSUE_PROFILES:
        keywords: Iterable[str] = profile["keywords"].get(lang) or profile["keywords"].get("en", ())
        if any(keyword in text for keyword in keywords):
            matches.append(profile)
    return matches


def fallback_car_details(description: str, language: str) -> str:
    strings = _strings(language)
    text = _normalise_text(description)

    brand_output = strings["unknown_value"]
    model_output = strings["unknown_value"]

    for key, info in _BRAND_PATTERNS.items():
        if key in text:
            brand_output = info["brand"]
            model_output = strings["unknown_value"]
            model_candidates: Iterable[str] = info.get("models", ())
            pattern = re.compile(rf"{re.escape(key)}\s+([a-z0-9\-]+)")
            match = pattern.search(text)
            if match:
                model_output = match.group(1).upper()
            else:
                for model in model_candidates:
                    if model in text:
                        model_output = model.upper()
                        break
            break

    lines = [
        f"{strings['car_details_header']}:",
        f"- {strings['brand_label']}: {brand_output}",
        f"- {strings['model_label']}: {model_output}",
        f"- {strings['engine_label']}: {strings['unknown_value']}",
        f"- {strings['transmission_label']}: {strings['unknown_value']}",
        f"- {strings['year_label']}: {strings['unknown_value']}",
    ]
    return "\n".join(lines)


def _extract_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"[.!?\n]", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def fallback_behaviors(description: str, language: str) -> str:
    strings = _strings(language)
    matches = _detect_profiles(description, language)

    if matches:
        lines: List[str] = [f"{strings['behaviors_header']}:"]
        seen: List[str] = []
        lang = _normalise_language(language)
        for profile in matches:
            behaviours: Iterable[str] = profile["behaviors"].get(lang) or profile["behaviors"].get("en", ())
            for entry in behaviours:
                if entry not in seen:
                    seen.append(entry)
        lines.extend(seen)
        return "\n".join(lines)

    sentences = _extract_sentences(description)
    text = _normalise_text(description)
    lang = _normalise_language(language)
    keywords: Iterable[str] = tuple({kw for profile in _ISSUE_PROFILES for kw in profile["keywords"].get(lang, ())})

    for sentence in sentences:
        lowered = _normalise_text(sentence)
        if any(keyword in lowered for keyword in keywords):
            return "\n".join([f"{strings['behaviors_header']}:", f"- {sentence.strip()}"])

    return ""


def fallback_changed_parts(description: str, language: str) -> str:
    strings = _strings(language)
    text = _normalise_text(description)
    lang = _normalise_language(language)

    triggers = _REPLACED_TRIGGERS.get(lang, ()) or _REPLACED_TRIGGERS.get("en", ())
    if not any(trigger in text for trigger in triggers):
        return ""

    parts: List[str] = []
    for profile in _ISSUE_PROFILES:
        entries: Iterable[Dict[str, object]] = profile["replaced_parts"].get(lang) or profile["replaced_parts"].get("en", ())
        for entry in entries:
            keywords: Iterable[str] = entry.get("keywords", ())
            if any(keyword in text for keyword in keywords):
                label = str(entry.get("label", "")).strip()
                if label and label not in parts:
                    parts.append(label)

    if not parts:
        return ""

    lines = [f"{strings['changed_parts_header']}:"]
    lines.extend(f"- {part}" for part in parts)
    return "\n".join(lines)


def fallback_possible_causes(state: Dict[str, str], language: str) -> str:
    strings = _strings(language)
    description = state.get("description_text", "")
    matches = _detect_profiles(description, language)
    if not matches:
        return ""

    lang = _normalise_language(language)
    lines = [f"{strings['possible_causes_header']}:"]
    seen: List[str] = []
    for profile in matches:
        causes: Iterable[str] = profile["causes"].get(lang) or profile["causes"].get("en", ())
        for cause in causes:
            if cause not in seen:
                seen.append(cause)
    lines.extend(seen)
    return "\n".join(lines)


def fallback_possible_solutions(state: Dict[str, str], language: str) -> str:
    strings = _strings(language)
    description = state.get("description_text", "")
    matches = _detect_profiles(description, language)
    if not matches:
        return ""

    lang = _normalise_language(language)
    mechanic_steps: List[str] = []
    user_advice: List[str] = []
    for profile in matches:
        mech_entries: Iterable[str] = profile["mechanic_steps"].get(lang) or profile["mechanic_steps"].get("en", ())
        user_entries: Iterable[str] = profile["user_advice"].get(lang) or profile["user_advice"].get("en", ())
        for step in mech_entries:
            if step not in mechanic_steps:
                mechanic_steps.append(step)
        for tip in user_entries:
            if tip not in user_advice:
                user_advice.append(tip)

    if not mechanic_steps and not user_advice:
        return ""

    lines: List[str] = [f"{strings['possible_solutions_header']}:"]
    if mechanic_steps:
        lines.append(f"{strings['mechanic_instructions_header']}:")
        lines.extend(f"- {step}" for step in mechanic_steps)
    if user_advice:
        lines.append(f"{strings['user_advice_header']}:")
        lines.extend(f"- {tip}" for tip in user_advice)

    return "\n".join(lines)
