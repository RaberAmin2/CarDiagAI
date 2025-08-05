from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def chat_node(state):
    llm = ChatOllama(model=bots["agent_chat"], base_url="http://localhost:11434", temperature=0)
    prompt = f"""
    You are a car mecanical AI diagnostic assistant. This desciption is provided to help you understand the context of the user's question:
    {json.dumps(state['description_text'], indent=2)}

    And this Text is the Output of the AI agent that is supposed to detect the car issues through logic and reasoning:
    {json.dumps(state['possible_causes'], indent=2)}

    With the above information,try to answer the user's question.Respond conversationally with insights.keep your response brief: 
    {state['user_question']}

   
    {{ "chat_response": "Your response here" }}
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content
        try:
            parsed = json.loads(result.strip())
            response = parsed.get("chat_response", result.strip())
        except json.JSONDecodeError:
            response = result.strip()
        chat_entry = {"question": state['user_question'], "response": response}
        chat_history = state.get('chat_history', []) + [chat_entry]
        return {"chat_response": response, "chat_history": chat_history}
    except Exception as e:
        return {"chat_response": "", "warning": str(e)}