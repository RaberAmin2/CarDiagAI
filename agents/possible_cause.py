from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def possible_cause(state):
    llm = ChatOllama(
        model=bots["possible_cause_agent"],
        base_url="http://localhost:11434",
        temperature=0.5
    )

    prompt = f"""
    Task: Suggest one or more possible technical causes of the reported problem based strictly on the provided information.
    - Consider car details, affected behaviors, noises, and changed parts.
    - Do not invent information that is not mentioned or clearly inferable.
    - Normalize all parts to standard automotive terms.
    - Do not list replaced parts as possible causes unless they are explicitly still suspected to be faulty.
    - Always respond in the same language as the input.

    Car Info: {json.dumps(state['car_details'], indent=2)}
    Problem Description: {state['description_text']}
    Affected Behaviors: {json.dumps(state['affected_behaviors'], indent=2)}
    Noises: {json.dumps(state['noises'], indent=2)}
    Changed Parts: {json.dumps(state['changed_parts'], indent=2)}

    Response format (no explanations outside the structure):
    POSSIBLE_CAUSES:
    - cause → reason

    If multiple causes are possible, list them in separate lines.
    If no possible causes can be derived, respond exactly with:
    POSSIBLE_CAUSES: None
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"possible_causes": result}
    except Exception as e:
        return {"possible_causes": "", "warning": str(e)}


