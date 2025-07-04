from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def noise(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt = f"""
    Using the following description,can you identify alle informations according to the (if described) niose? all information is in the form of a list of strings. 
    {json.dumps(state['description_text'], indent=2)}

    Noise sound, pattern,frequency, and any other relevant details should be included in the response.
    if no noise is described, return a list which contain a "none" element.
    if you are unsure about any of the information, add a "maybe"
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"noises": result.strip()}
    except Exception as e:
        return {"noises": "", "warning": str(e)}