from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def possible_cause(state):
    llm = ChatOllama(
        model=bots["possible_cause_agent"],
        base_url="http://localhost:11434",
        temperature=0
    )

    prompt = f"""
    Task: Identify possible causes of the problem based only on the provided information. 
    For each cause, provide a short justification as a mechanic would.

    Car Info: {json.dumps(state['car_details'], indent=2)}
    Problem Description: {state['description_text']}
    Affected Behaviors: {json.dumps(state['affected_beaviors'], indent=2)}
    Noises: {json.dumps(state['noises'], indent=2)}
    Changed Parts: {json.dumps(state['changed_parts'], indent=2)}

    Response format (no explanations outside the structure):
    POSSIBLE_CAUSES:
    - cause 1 → reason 1
    - cause 2 → reason 2
    - cause 3 → reason 3

    If there is not enough information:
    POSSIBLE_CAUSES: None
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"possible_causes": result}
    except Exception as e:
        return {"possible_causes": "", "warning": str(e)}
