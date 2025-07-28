from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def possible_cause(state):
    llm = ChatOllama(model=bots["possible_cause_agent"], base_url="http://localhost:11434")
    prompt = f"""
    According to the following Descriptions of the car identify the possible causes of the problem by thinking like a mechanic.
    Car Info:{json.dumps(state['car_details'], indent=2)}
    Problem Description: {state['description_text']}
    Affected Behaviors: {json.dumps(state['affected_beaviors'], indent=2)}
    Noises: {json.dumps(state['noises'], indent=2)}
    Changed Parts: {json.dumps(state['changed_parts'], indent=2)}
    Respond with a detailed text that includes the most likly causes of the problem.
    Even if the did not provide all the information about the car or the problem, try to identify the possible causes based on the available information.
    But do not make assumptions about the car or the problem, only use the provided information. You can use your knowledge about cars to identify the possible causes.
    If you cannot identify any possible causes, respond with 'Not enough information to identify the possible causes'.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"possible_causes": result.strip()}
    except Exception as e:
        return {"possible_causes": "", "warning": str(e)}