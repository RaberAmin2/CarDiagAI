from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

def identify_car(state):
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
    prompt = f"""
    Using the following description,can you identify the car brand, the model and the possible engine, the possible transmission and the manufacturing year?
    {json.dumps(state['description_text'], indent=2)}

    If you can identify the requested information, or just parts of it, please return a list of dictionaries with the keys "brand", "model", "engine", "transmission", and "year".
    If you cannot identify any of the requested information, return an empty list.
    If you are unsure about any of the information, you can fill in possible informations.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"car_details": result.strip()}
    except Exception as e:
        return {"car_details": "", "warning": str(e)}