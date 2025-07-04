from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def behavior(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt = f"""
    Using the following description,can you identify all informations about the misbehavior of the car? all information is in the form of a list of strings. 
    {json.dumps(state['description_text'], indent=2)}

    Every pice of information could be important for the diagnosis, so please include all relevant details.
    If you cannot identify any of the requested information, return a list with "none" as the only element.
    If you are unsure about any of the information, add a "maybe"
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"affected_beaviors": result.strip()}
    except Exception as e:
        return {"affected_beaviors": "", "warning": str(e)}