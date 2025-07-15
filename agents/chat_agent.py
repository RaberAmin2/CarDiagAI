from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json

def chat_node(state):
    llm = ChatOllama(model="llava-llama3", base_url="http://localhost:11434")
    prompt = f"""
    Context:
    description: {json.dumps(state['description_text'], indent=2)}

    User Question:
    {state['user_question']}

    Respond conversationally with insights or suggestions : keep your response brief
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