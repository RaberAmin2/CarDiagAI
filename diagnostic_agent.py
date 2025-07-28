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
    car_details: str
    affected_parts: str
    affected_beaviors: str
    possible_causes: str
    possible_solutions: str
    noises: str
    changed_parts: str
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
    st.session_state.state = {
        "description_text": "",
        "affected_parts": [],
        "affected_beaviors": [],
        "possible_causes": [],
        "possible_solutions": [],
        "noises": [],
        "car_details": [],
        "changed_parts": [],
        "chat_response": "",
        "user_question": "",
        "chat_history": [],
    }

with st.form("diagnostic_form"):

    col1= st.columns(1)
    if col1:
        a=st.text_area("Beschreibung des Problems")
    #Verarbeite die Eingaben
    submit_btn = st.form_submit_button("Diagnose starten")

if submit_btn:    
    #problem_summary={"description_text":description_text}
    st.session_state.state.update({
        "chat_history": "",
        "user_question": "",
        "chat_response": "",
        "car_details":"",
        "description_text":a,
        "affected_parts": "",
        "affected_beaviors": "",
        "possible_causes": "",
        "possible_solutions": "",
        "noises": "",
        "changed_parts": "",
    })
    with st.spinner("Generating Diagnosis..."):
        result = graph.invoke(st.session_state.state)
        st.session_state.state.update(result)
        if result.get("possible_solutions"):
            st.success("Diagnosis Created")
        else:
            st.error("Failed to generate Diagnosis.")
#here########################
#layout
if st.session_state.state.get("possible_solutions") :
    col_itin, col_chat = st.columns([3, 2])

    with col_itin:
        st.markdown("### Diagnosed Car Problem")
        st.markdown(st.session_state.state["possible_solutions"])

        # # All agent buttons in one row
        # col_btn1, col_btn2, col_btn3, col_btn4, col_btn5,col_btn6 = st.columns(6)
        # with col_btn1:
        #     if st.button("Get Activity Car-info"):
        #         with st.spinner("Fetching Car Details..."):
        #             result = identify_car.identify_car(st.session_state.state)
        #             st.session_state.state.update(result)
        # with col_btn2:
        #     if st.button("Identify Missbehaviur"):
        #         with st.spinner("Detecting missbehaviour..."):
        #             result = behavior.behavior(st.session_state.state)
        #             st.session_state.state.update(result)
        # with col_btn3:
        #     if st.button("Analyzing noise"):
        #         with st.spinner("Analyzing noise Description..."):
        #             result = noise.noise(st.session_state.state)
        #             st.session_state.state.update(result)
        # with col_btn4:
        #     if st.button("New Parts"):
        #         with st.spinner("Thinkging about new previously installed Parts..."):
        #             result = new_parts.new_parts(st.session_state.state)
        #             st.session_state.state.update(result)
        # with col_btn5:
        #     if st.button("Analyzing Possible Cause"):
        #         with st.spinner("Analyzing Causes..."):
        #             result = possible_cause.possible_cause(st.session_state.state)
        #             st.session_state.state.update(result)
        # with col_btn6:
        #     if st.button("Analyzing Possible Solutions"):
        #         with st.spinner("Analyzing Solutions..."):
        #             result = possible_solution.possible_solution(st.session_state.state)
        #             st.session_state.state.update(result)

        # Display all agent outputs in expanders
        if st.session_state.state.get("Car Details"):
            with st.expander("Details of the Car", expanded=False):
                st.markdown(st.session_state.state["car_details"])

        if st.session_state.state.get("affected_parts"):
            with st.expander("Faulty Parts", expanded=False):
                st.markdown(st.session_state.state["affected_parts"])

        if st.session_state.state.get("possible Causes"):
            with st.expander("Identifyed Problem", expanded=False):
                st.markdown(st.session_state.state["possible_causes"])

        if st.session_state.state.get("Solution"):
            with st.expander("possible Solutions", expanded=False):
                st.markdown(st.session_state.state["possible_solutions"])

        # Export PDF button
        if st.button("Export as PDF"):
            pdf_path = export_to_pdf("The Users description was : "+st.session_state.state["description_text"]+" Agent for car Misbehaviur extracted : "+st.session_state.state["affected_beaviors"]+" Agent for noise extractain said : "+st.session_state.state["noises"]+" Agent for Replaced Parts extracted : "+st.session_state.state["changed_parts"]+"Agent for car information said : "+st.session_state.state["car_details"]+"Agent for possible causes said : "+st.session_state.state["possible_causes"]+"Agent for possible solutions said : "+st.session_state.state["possible_solutions"],)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button("Download Diagnosis PDF", f, file_name="Diagnosis.pdf")

    with col_chat:
        st.markdown("### Chat About Your Problem")
        for chat in st.session_state.state["chat_history"]:
            with st.chat_message("user"):
                st.markdown(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(chat["response"])

        if user_input := st.chat_input("Ask something about your Problem"):
            st.session_state.state["user_question"] = user_input
            with st.spinner("Generating response..."):
                result = chat_agent.chat_node(st.session_state.state)
                st.session_state.state.update(result)
                st.rerun()
else:
    st.info("")
                    
            