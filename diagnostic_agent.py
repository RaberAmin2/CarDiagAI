#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:50:07 2025
Programm zur Diagnose von Fahrzeugproblemen mittels KI.
DiaKari ist ein KI-gestützter Fahrzeugdiagnose-Agent, der auf der Llama3.2 basiert.
@author: razam
"""
import streamlit as st
import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import os
from agents import identify_car,new_parts,noise,behavior,possible_solution,chat_agent,possible_cause
from utils_export import export_to_pdf

#lädt umgebungs variabeln
load_dotenv()

#llm init
st.set_page_config(page_title="AI Car Diag",layout="wide",page_icon="icon.png")
try:
    llm=ChatOllama(model="llama3.2",base_url="http://localhost:11434")
except Exception as e:
    st.error(f"LLm init failed: {str(e)}")
    st.stop()

#skipped serper
    
"""
Beschreibung des Prolems in textform.
Beinflussung auf das fahrzeug bzw fahrverhalten.
Mögliche Ursachen und Lösungen.
Geräusche die auftreten.
Geänderte Teile des Fahrzeugs.
"""
#Definiere status
class GraphState(TypedDict):
    description_text: str
    car_details: list[str]
    affected_parts: list[str]
    affected_beaviors: list[str]
    possible_causes: list[str]
    possible_solutions: list[str]
    noises: list[str]
    changed_parts: list[str]
    chat_response: str
    user_question: str
    chat_history: Annotated[list[dict],"Chat history for the conversation"]

#Langgraph workflow
workflow = StateGraph(GraphState)
workflow.add_node("identify_car",identify_car.identify_car)
workflow.add_node("new_parts",new_parts.new_parts)
workflow.add_node("noise",noise.noise)
workflow.add_node("behavior",behavior.behavior)
workflow.add_node("possible_solution",possible_solution.possible_solution)
workflow.add_node("possible_cause",possible_cause.possible_cause)
workflow.add_node("chat",chat_agent.chat_node)
    
##Init Defined Workflow !!!!!!!!!!!!!!!!!!!!
workflow.set_entry_point("identify_car")
workflow.add_edge("identify_car", "behavior")
workflow.add_edge("behavior", "noise")
workflow.add_edge("noise", "new_parts")
workflow.add_edge("new_parts", "possible_cause")
workflow.add_edge("possible_cause", "possible_solution")
workflow.add_edge("possible_solution", "chat")
workflow.add_edge("chat", END)
graph=workflow.compile()

#ui
st.markdown("# AI Car Diagnostic Agent")

if 'state' not in st.session_state:
    st.session_state.state={
        "description_text": "",
        "affected_parts": [],
        "affected_beaviors": [],
        "possible_causes": [],
        "possible_solutions": [],
        "noises": [],
        "changed_parts": [],
        "chat_response": "",
        "user_question": "",
        "chat_history": [],
    }

with st.form("diagnostic_form"):
    col1,col2= st.columns(2)
    with col1:
        st.text_area("Beschreibung des Problems", key="description_text", placeholder="Beschreiben Sie das Problem Ihres Fahrzeugs... Bitte geben Sie auch das fahrzeug inklusive Baujahr an.", height=200)
        st.text_input("Betroffene Teile", key="affected_parts", placeholder="Z.B. Motor, Getriebe, etc.")
        st.text_input("Betroffene Verhaltensweisen", key="affected_beaviors", placeholder="Z.B. Leistungsverlust, Geräusche, etc.")
        st.text_input("Mögliche Ursachen", key="possible_causes", placeholder="Z.B. Verschleiß, Defekt, etc.")
        st.text_input("Mögliche Lösungen", key="possible_solutions", placeholder="Z.B. Reparatur, Austausch, etc.")
        st.text_input("Geräusche", key="noises", placeholder="Z.B. Klopfen, Quietschen, etc.")
        st.text_input("Geänderte Teile des Fahrzeugs", key="changed_parts", placeholder="Z.B. Reifen, Bremsen, etc.")
    
    submit_button = st.form_submit_button(label='Diagnose starten')
    #Verarbeite die Eingaben
if submit_button:    
    problem_summary_text= f"Beschreibung des Problems: {description_text}\nBetroffene Teile: {affected_parts}\nBetroffene Verhaltensweisen: {affected_beaviors}\nMögliche Ursachen: {possible_causes}\nMögliche Lösungen: {possible_solutions}\nGeräusche: {noises}\nGeänderte Teile des Fahrzeugs: {changed_parts}"
    problem_summary={"Beschreibung des Problems":description_text,
    "Betroffene Teile": affected_parts,
    "Betroffene Verhaltensweisen": affected_beaviors,
    "Mögliche Ursachen": possible_causes,
    "Mögliche Lösungen": possible_solutions,
    "Geräusche": noises,
    "Geänderte Teile des Fahrzeugs": changed_parts
    }
     
    st.session_state.state.update({
        "problem_summary_text": "",
        "problem_summary": "",
        "chat_history": "",
        "user_question": "",
        "chat_response": "",
        "description_text":"" ,
        "affected_parts": "",
        "affected_beaviors": "",
        "possible_causes": "",
        "possible_solutions": "",
        "noises": "",
        "changed_parts": "",
    })

#layout
if st.session_state.state.get("problem_summary_text"):
    col_description, col_chat = st.columns(0,1)
    col_btn1,col_btn2,col_btn3,col_btn4 = st.columns(4)
    with col_btn1:
        if st.button("Diagnose starten"):
            with st.spinner("Diagnose wird durchgeführt..."):
                result=description_text.description_text(st.session_state.state)
                st.session_state.upade(result)

    with col_btn2:
        if st.button("possible_causes"):
            with st.spinner("Mögliche Ursachen werden ermittelt..."):
                result=possible_causes.possible_causes(st.session_state.state)
                st.session_state.update(result)
    
    with col_btn3:
        if st.button("possible_solutions"):
            with st.spinner("Mögliche Lösungen werden ermittelt..."):
                result=possible_solutions.possible_solutions(st.session_state.state)
                st.session_state.update(result)
    
    with col_btn4:
        if st.button("Exportieren"):
            with st.spinner("Exportiere Diagnose..."):
                pdf_path=export_to_pdf(st.session_state.state["problem_summary", "chat_history","noises","changed_parts","affected_parts","affected_beaviors","possible_causes","possible_solutions"])
                if pdf_path:
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="Download Diagnose PDF",
                            data=pdf_file,
                            file_name="diagnose_report.pdf",
                            mime="application/pdf"
                        )
                st.success("Diagnose erfolgreich exportiert!")

    with col_chat:
        st.markdown("## Chat")
        for chat in st.session_state.state["chat_history"]:
            with st.chat_message("user"):
                st.markdown(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(chat["response"])
        
        if user_input:= st.session_state.state["Frage etwas"] :
            st.session_state.state["user_question"]= user_input
            with st.spinner("Antwort wird generiert..."):
                result = chat_agent.chat_node(st.session_state.state)
                st.session_state.state.update(result)
                st.rerun()
else:
    st.info("Bitte füllen Sie das Formular aus, um eine Diagnose zu starten.")
                    
            