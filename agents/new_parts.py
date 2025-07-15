from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def new_parts(state):
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    prompt = f"""
    Using the following description,extract the information about new parts that may have been replaced in the car.  
    {json.dumps(state['description_text'], indent=2)}

   respond with a short text that includes the names of the new parts.
   if there are no new parts, respond with 'No new parts were described'.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"changed_parts": result.strip()}
    except Exception as e:
        return {"changed_parts": "", "warning": str(e)}