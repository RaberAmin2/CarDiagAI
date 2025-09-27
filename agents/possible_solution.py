from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def possible_solution(state):
    llm = ChatOllama(model=bots["possible_solution_agent"], base_url="http://localhost:11434")

    prompt = f"""
    Task: Based on the possible causes provided, generate a structured solution.
    - Provide clear step-by-step instructions for a mechanic to solve the issue.
    - Indicate if the user can safely perform any of the steps themselves (e.g., checking fluid levels, visually inspecting parts).
    - Base all instructions only on the given possible causes. Do not invent new causes.
    - Always respond in the same language as the input.

    Possible Causes:
    {json.dumps(state['possible_causes'], indent=2)}

    Response format (no extra explanations):
    POSSIBLE_SOLUTIONS:
    Mechanic instructions:
    - step 1
    - step 2
    - step 3

    User advice:
    - advice 1
    - advice 2

    If no possible causes are given, respond exactly with:
    POSSIBLE_SOLUTIONS: None
    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        return {"possible_solutions": result.strip()}
    except Exception as e:
        return {"possible_solutions": "", "warning": str(e)}
