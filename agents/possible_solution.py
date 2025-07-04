from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def possible_solution(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt = f"""
    Using the following information, can you identify the problem with the car? 
    {json.dumps(state['possible_causes'], indent=2)}

    If you can identify the problem add one of 3 "sureness levels" Sure,Possible,Maybe.
    If in addition to you diagnoses their are known problems with this type of car which can relate to this problem, mention them too.
    If you cannot identify the problem, return the string "Unknown Problem Code:4".
    
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"possible_solutions": result.strip()}
    except Exception as e:
        return {"possible_solutions": "", "warning": str(e)}