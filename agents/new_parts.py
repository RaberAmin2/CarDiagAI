from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def new_parts(state):
    llm = ChatOllama(model=bots["new_parts_agent"], base_url="http://localhost:11434", temperature=0)

    prompt = f"""
    Task: Extract only information about newly replaced parts from the following description.

    Description:
    {json.dumps(state['description_text'], indent=2)}

    Response format (no explanations, no extra text):
    NEW_PARTS: part1, part2, part3

    If no new parts are described:
    NEW_PARTS: None
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"changed_parts": result}
    except Exception as e:
        return {"changed_parts": "", "warning": str(e)}
