from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def noise(state):
    llm = ChatOllama(model=bots["noise_agent"], base_url="http://localhost:11434", temperature=0)

    prompt = f"""
    Task: Extract only noise- or sound-related information from the following user description. 
    Ignore all unrelated information. If no noise is described, respond with "NOISES: None".
    Always respond in the same language as the description. Do not translate.

    Description:
    {json.dumps(state['description_text'], indent=2)}

    Response format (no explanations, no extra text):
    NOISES:
    1. Sound: ...
       Pattern: ...
       Frequency: ...
       Details: ...

    If multiple noises are described, list them as 1, 2, 3, ...
    If no noises are described, respond exactly with:
    NOISES: None
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"noises": result}
    except Exception as e:
        return {"noises": "", "warning": str(e)}

