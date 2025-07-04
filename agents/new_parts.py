from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def new_parts(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt = f"""
    Using the following description,can you identify if the user have changed any parts of his car until this request?
    {json.dumps(state['description_text'], indent=2)}

    If you can identify the requested information, return a list of the changed parts.
    If you cannot identify any of the requested information, return a list with "none" as the only element.
    If you are unsure about any of the information, add a "maybe" to the elements of the list.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"changed_parts": result.strip()}
    except Exception as e:
        return {"changed_parts": "", "warning": str(e)}