from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def noise(state):
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    prompt = f"""
    Using the following description,can you identify all the noises that are present in the car which may indicate a problem? if the user tried to describe the noises, please extract the relevant information.
    {json.dumps(state['description_text'], indent=2)}

    respond with a short text that includes Noise sound, pattern,frequency, and any other relevant details.
    Do not include any other information or text, just the noises.Begin your response with 'Noises'.
    if there are no noises, respond with 'No noises were described'
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"noises": result.strip()}
    except Exception as e:
        return {"noises": "", "warning": str(e)}