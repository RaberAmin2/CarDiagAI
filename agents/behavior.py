# agents/behavior.py

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json
import logging

# Lade zentrale Modellkonfiguration
with open("agents/bots_settings.json") as f:
    bots = json.load(f)

# Optional: Logging aktivieren
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def behavior(state: dict) -> dict:
    try:
        llm = ChatOllama(
            model=bots.get("behavior_agent", "llama3"),
            base_url="http://localhost:11434",
            temperature=0
        )

        prompt = f"""
Using the following description, identify all information about the behavior of the car (e.g., shaking, vibrations, steering issues, braking issues, acceleration issues, etc.). Every piece of information could be important for the diagnosis.

Description:
{state.get('description_text', '')}

Answer in a short, clear text starting with 'Affected behaviors: ...'.  
If no behavior can be identified, reply with 'No affected behaviors identified'.  
Do not add explanations or extra formatting.
Answer in the Language of the input.

"""

        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

        logger.info(f"[Behavior Agent] Output: {result}")

        return {
            "affected_beaviors": result
        }

    except Exception as e:
        logger.error(f"[Behavior Agent] Error: {str(e)}")
        return {
            "affected_beaviors": "No affected behaviors identified.",
            "warning": str(e)
        }
