from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def identify_car(state):
    llm = ChatOllama(model=bots["identify_car_agent"], base_url="http://localhost:11434", temperature=0)

    prompt = f"""
    Task: Extract vehicle details from the following problem description.
    - Normalize model names and technical details (e.g., Golf VII → Golf 7).
    - If a detail can be reasonably inferred from the description (e.g., "Golf VII" → Brand: VW, Model: Golf 7), include it.
    - If a detail cannot be identified or inferred with certainty, write "Unknown".
    - Always respond in the same language as the description. Do not translate.

    Description:
    {json.dumps(state['description_text'], indent=2)}

    Response format (no extra words, no explanations):
    CAR_DETAILS:
    - Brand: ...
    - Model: ...
    - Engine: ...
    - Transmission: ...
    - Year: ...
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"car_details": result}
    except Exception as e:
        return {"car_details": "", "warning": str(e)}
