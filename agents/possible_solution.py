from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def possible_solution(state):
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    prompt = f"""
    Using the following information prepare a detaild text for a mechanic to solve the Problem.
    {json.dumps(state['possible_causes'], indent=2)}

    Respond with a detailed text to solve the problem.
    
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"possible_solutions": result.strip()}
    except Exception as e:
        return {"possible_solutions": "", "warning": str(e)}