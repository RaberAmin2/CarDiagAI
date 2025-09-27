from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json 

with open('agents/bots_settings.json') as f:
    bots = json.load(f)

def noise(state):
    llm = ChatOllama(model=bots["noise_agent"], base_url="http://localhost:11434", temperature=0)

    prompt = f"""
    Task: Extract only the noise-related information from the following description.

    Description:
    {json.dumps(state['description_text'], indent=2)}

    Required data:
    - Noise type / sound
    - Pattern or progression
    - Frequency
    - Other relevant details

    Response format (no explanations, no extra text):
    NOISES: {{sound}}, {{pattern}}, {{frequency}}, {{details}}

    If no noises are described:
    NOISES: None
    Answer in the Language of the input.

    """

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        return {"noises": result}
    except Exception as e:
        return {"noises": "", "warning": str(e)}
