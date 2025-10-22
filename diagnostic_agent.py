#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:50:07 2025
Programm zur Diagnose von Fahrzeugproblemen mittels KI.
DiaKari ist ein KI-gestützter Fahrzeugdiagnose-Agent, der auf der Llama3.1 basiert.
@author: razam
"""
import streamlit as st
import json
from typing import Any, Dict, Iterable, TypedDict
from langgraph.graph import StateGraph, END
import logging
from agents import (
    identify_car,
    new_parts,
    noise,
    behavior,
    possible_solution,
    chat_agent,
    possible_cause,
)
from utils_export import export_to_pdf
import requests


# Logging konfigurieren
logging.basicConfig(
    filename="diagnostic_agent.log",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("\U0001f680 DiaKari Diagnostic Agent gestartet")

# UI Setup
st.set_page_config(page_title="AI Car Diag", layout="wide", page_icon="icon.png")

# Testmodus
debug_mode = True
if debug_mode:
    with open("test_text.txt", "r") as file:
        test_text = file.read().strip()
else:
    test_text = ""


# Statusdefinition
class GraphState(TypedDict, total=False):
    """Shared state passed between the different LangGraph nodes."""

    description_text: str
    car_details: str
    affected_parts: str
    affected_behaviors: str
    possible_causes: str
    possible_solutions: str
    noises: str
    changed_parts: str
    chat_response: str
    user_question: str
    chat_history: list[dict[str, str]]


# Langgraph Workflow
workflow = StateGraph(GraphState)
workflow.add_node("identify_car", identify_car.identify_car)
workflow.add_node("new_parts", new_parts.new_parts)
workflow.add_node("noise", noise.noise)
workflow.add_node("behavior", behavior.behavior)
workflow.add_node("possible_solution", possible_solution.possible_solution)
workflow.add_node("possible_cause", possible_cause.possible_cause)
workflow.add_node("chat", chat_agent.chat_node)
# workflow.add_node("stop_models", stop_models_node)


workflow.set_entry_point("identify_car")
workflow.add_edge("identify_car", "behavior")
workflow.add_edge("behavior", "noise")
workflow.add_edge("noise", "new_parts")
workflow.add_edge("new_parts", "possible_cause")
workflow.add_edge("possible_cause", "possible_solution")
workflow.add_edge("possible_solution", "chat")
workflow.add_edge("chat", END)
# workflow.add_edge("chat", "stop_models")
# workflow.add_edge("stop_models", END)
graph = workflow.compile()


AGENT_SEQUENCE = (
    (identify_car.identify_car, ("car_details",)),
    (behavior.behavior, ("affected_behaviors",)),
    (noise.noise, ("noises",)),
    (new_parts.new_parts, ("changed_parts",)),
    (possible_cause.possible_cause, ("possible_causes",)),
    (possible_solution.possible_solution, ("possible_solutions",)),
)


def run_diagnosis_pipeline(
    state: Dict[str, Any], locked_fields: Iterable[str] | None = None
) -> Dict[str, Any]:
    """Recalculate the diagnosis when new information is provided via chat."""

    working_state: Dict[str, Any] = dict(state)
    locked = set(locked_fields or [])
    aggregated_updates: Dict[str, Any] = {}

    for agent_fn, produced_keys in AGENT_SEQUENCE:
        agent_result = agent_fn(working_state)
        working_state.update(agent_result)

        for key in produced_keys:
            if key in agent_result and key not in locked:
                aggregated_updates[key] = agent_result[key]

    return aggregated_updates

# UI Start
st.markdown("# AI Car Diagnostic Agent")

if "state" not in st.session_state:
    logging.info("🔄 Session State wird initialisiert.")
    st.session_state.state = {
        "description_text": "",
        "affected_parts": "",
        "affected_behaviors": "",
        "possible_causes": "",
        "possible_solutions": "",
        "noises": "",
        "car_details": "",
        "changed_parts": "",
        "chat_response": "",
        "user_question": "",
        "chat_history": [],
    }

a = ""

with st.form("diagnostic_form"):
    col1 = st.columns(1)[0]
    a = st.text_area("Beschreibung des Problems", value=test_text)
    submit_btn = st.form_submit_button("Diagnose starten")

if submit_btn:
    logging.info("📝 Diagnose wurde gestartet.")
    if not a.strip():
        logging.warning("⚠️ Leere Eingabe im Textfeld. Diagnose abgebrochen.")
        st.warning("Bitte gib eine Beschreibung des Problems ein.")
        st.stop()

    st.session_state.state.update(
        {
            "chat_history": [],
            "user_question": "",
            "chat_response": "",
            "car_details": "",
            "description_text": a,
            "affected_parts": "",
            "affected_behaviors": "",
            "possible_causes": "",
            "possible_solutions": "",
            "noises": "",
            "changed_parts": "",
        }
    )
    logging.debug(f"📅 Eingabebeschreibung: {a}")

    with st.spinner("Generating Diagnosis..."):
        try:
            result = graph.invoke(st.session_state.state)
            st.session_state.state.update(result)
            logging.info("✅ Diagnose erfolgreich generiert.")
            logging.debug(f"📊 Diagnosedaten: {json.dumps(result, indent=2)}")

            if result.get("possible_solutions"):
                st.success("Diagnosis Created")
            else:
                st.error("Failed to generate Diagnosis.")
                logging.warning("⚠️ Keine Lösungsvorschläge gefunden.")
        except Exception as e:
            logging.exception(f"❌ Fehler bei der Graph-Ausführung: {e}")
            st.error("Fehler bei der Diagnoseausführung.")

if st.session_state.state.get("possible_solutions"):
    col_itin, col_chat = st.columns([3, 2])
    with col_itin:
        st.markdown("### 🧠 Diagnose")
        st.markdown("#### 🚘 Fahrzeuginfo")
        st.markdown(f"> {st.session_state.state['car_details']}")

        st.markdown("#### 💠 Erkanntes Fehlverhalten")
        st.markdown(f"> {st.session_state.state['affected_behaviors']}")

        st.markdown("#### 🔊 Geräusche")
        st.markdown(f"> {st.session_state.state['noises']}")

        st.markdown("#### 🧹 Erkannte defekte Teile")
        st.markdown(f"> {st.session_state.state['affected_parts']}")

        st.markdown("#### 🔄 Ersetzte Teile")
        st.markdown(f"> {st.session_state.state['changed_parts']}")

        st.markdown("#### ❓ Mögliche Ursachen")
        for line in st.session_state.state["possible_causes"].split("\n"):
            st.markdown(f"- {line}")

        st.markdown("#### 💠 Lösungsvorschläge")
        for line in st.session_state.state["possible_solutions"].split("\n"):
            st.markdown(f"- {line}")

        if st.button("📄 Diagnose als PDF exportieren"):
            try:
                pdf_path = export_to_pdf(
                    "Die Nutzereingabe war: "
                    + st.session_state.state["description_text"]
                    + "\n\nFehlverhalten: "
                    + st.session_state.state["affected_behaviors"]
                    + "\n\nGeräusche: "
                    + st.session_state.state["noises"]
                    + "\n\nErsetzte Teile: "
                    + st.session_state.state["changed_parts"]
                    + "\n\nFahrzeugdetails: "
                    + st.session_state.state["car_details"]
                    + "\n\nMögliche Ursachen: "
                    + st.session_state.state["possible_causes"]
                    + "\n\nLösungen: "
                    + st.session_state.state["possible_solutions"]
                )
                logging.info(f"📄 PDF-Export erfolgreich: {pdf_path}")
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📅 PDF herunterladen", f, file_name="Diagnose.pdf"
                    )
            except Exception as e:
                logging.error(f"❌ PDF-Export fehlgeschlagen: {e}")
                st.error("Fehler beim PDF-Export.")

        if st.checkbox("🔧 Manuellen Agenten-Modus aktivieren"):
            st.markdown("Agenten manuell ausführen:")
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            with col_btn1:
                if st.button("🚘 Fahrzeuginfo"):
                    logging.info("🔍 Manueller Agentenaufruf: identify_car")
                    with st.spinner("Fahrzeuginfo wird geladen..."):
                        result = identify_car.identify_car(st.session_state.state)
                        st.session_state.state.update(result)
                        logging.debug(f"🔀 Ergebnis von identify_car: {result}")
            with col_btn2:
                if st.button("📈 Fehlverhalten analysieren"):
                    logging.info("🔍 Manueller Agentenaufruf: behavior")
                    with st.spinner("Analysiere Verhalten..."):
                        result = behavior.behavior(st.session_state.state)
                        st.session_state.state.update(result)
                        logging.debug(f"🔀 Ergebnis von behavior: {result}")
            with col_btn3:
                if st.button("🔊 Geräusche analysieren"):
                    logging.info("🔍 Manueller Agentenaufruf: noise")
                    with st.spinner("Analysiere Geräusche..."):
                        result = noise.noise(st.session_state.state)
                        st.session_state.state.update(result)
                        logging.debug(f"🔀 Ergebnis von noise: {result}")

    with col_chat:
        st.markdown("### 💬 Chat zur Diagnose")
        for chat in st.session_state.state["chat_history"]:
            with st.chat_message("user"):
                st.markdown(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(chat["response"])

        if user_input := st.chat_input("Frage etwas zur Diagnose..."):
            logging.info(f"💬 Neue Benutzerfrage: {user_input}")
            st.session_state.state["user_question"] = user_input
            with st.spinner("Antwort wird generiert..."):
                result = chat_agent.chat_node(st.session_state.state)
                logging.debug(f"🤖 Chat-Agent Antwort: {result.get('chat_response')}")

                locked_fields = set(result.pop("locked_fields", []))
                regenerate = bool(result.pop("regenerate", False))

                st.session_state.state.update(result)
                st.session_state.state["user_question"] = ""

                if regenerate:
                    try:
                        logging.info(
                            "🔁 Zusätzliche Informationen erkannt – Diagnose wird aktualisiert."
                        )
                        diagnosis_updates = run_diagnosis_pipeline(
                            st.session_state.state, locked_fields
                        )
                        st.session_state.state.update(diagnosis_updates)
                        logging.debug(
                            "🧮 Aktualisierte Diagnosefelder: %s",
                            json.dumps(diagnosis_updates, indent=2, ensure_ascii=False),
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        logging.exception(
                            "❌ Fehler beim Aktualisieren der Diagnose nach Chat-Eingabe: %s",
                            exc,
                        )

                st.rerun()
else:
    logging.info("ℹ️ Kein Text eingegeben. Warte auf Benutzereingabe.")
    st.info("Bitte gib eine Problembeschreibung ein und starte die Diagnose.")


def stop_ollama_models():
    try:
        response = requests.post("http://localhost:11434/api/stop")
        print(f"🛑 Ollama stop response: {response.status_code}")
    except Exception as e:
        print(f"❌ Fehler beim Modell-Stop: {e}")


def stop_models_node(state):
    stop_ollama_models()
    return state
