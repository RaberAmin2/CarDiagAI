from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def identify_car(state):
    llm = ChatOllama(model=bots["identify_car_agent"], base_url="http://localhost:11434", temperature=0)

    prompt = f"""
    Task: Extract only vehicle details from the following problem description.

    Description:
    {json.dumps(state['description_text'], indent=2)}

    Required Data:
    - Brand
    - Model
    - Engine
    - Transmission
    - Year

    Response format (no extra words, no explanations):
    CAR_DETAILS: {{brand}}, {{model}}, {{engine}}, {{transmission}}, {{year}}

    If the details cannot be identified:
    CAR_DETAILS: Unknown
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"car_details": result}
    except Exception as e:
        return {"car_details": "", "warning": str(e)}
