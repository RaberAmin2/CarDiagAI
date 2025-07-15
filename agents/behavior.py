from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def behavior(state):
    llm = ChatOllama(model="mistral", base_url="http://localhost:11434")
    prompt = f"""
    Using the following description, identify all informations about the behavior of the car. like shaking, vibrations, steering issues, braking issues, acceleration issues, etc.Every pice of information could be important for the diagnosis.
    {json.dumps(state['description_text'], indent=2)}

    Answer in the in a short detailed text describing the affected behaviors of the car an when they occur.Begin with 'Affected behaviors' 
    If you cannot identify any behaviors, respond with 'No affected behaviors identified'.
    Do not include any additional text or explanations, just the affected behaviors.

    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"affected_beaviors": result.strip()}
    except Exception as e:
        return {"affected_beaviors": "", "warning": str(e)}