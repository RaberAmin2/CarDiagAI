from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def possible_cause(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt = f"""
    According to the following problem description, can you describe the possible solutions for the car?
     Car Info:{json.dumps(state['car_details'], indent=2)}
    Problem Description: {state['description_text']}
    Affected Parts: {json.dumps(state['affected_parts'], indent=2)}
    Affected Behaviors: {json.dumps(state['affected_beaviors'], indent=2)}
    Noises: {json.dumps(state['noises'], indent=2)}
    Changed Parts: {json.dumps(state['changed_parts'], indent=2)}
    If you could tell the solution also tell if it is possible to fix it yourself or if you need a mechanic.
    If you cannot identify a solution, return the string "Unknown Solution Code:4".
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"possible_causes": result.strip()}
    except Exception as e:
        return {"possible_causes": "", "warning": str(e)}