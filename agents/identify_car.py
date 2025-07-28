from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def identify_car(state):
    llm = ChatOllama(model=bots["identify_car_agent"], base_url="http://localhost:11434"temperature=0)
    prompt = f"""
    Using the following description,can you identify the car brand, the model and the engine, the  transmission and the manufacturing year?
    If the description is not sufficient, try to extract the necessary details from the text for example which car model was when produced, which engine was used, which transmission was used and which year it was produced.
    {json.dumps(state['description_text'], indent=2)}

    If you can identify the required information, answer in the following format without any additional text:
    'CAR_DETAILS: {{brand}}, {{model}}, {{engine}}, {{transmission}}, {{year}}'
    If you cannot identify the car, respond with 'CAR_DETAILS: Unknown'.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"car_details": result.strip()}
    except Exception as e:
        return {"car_details": "", "warning": str(e)}