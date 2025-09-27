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
Extract only information about the car's behavior from the following description. 
This includes driving dynamics and performance issues (e.g., shaking, vibrations, steering problems, braking issues, acceleration issues, stalling, pulling, loss of power). 
Ignore noises, replaced parts, or vehicle specifications.

Normalize the behaviors into clear, concise automotive terms. 
If the user uses informal phrases, rewrite them as standard behavior descriptions.

Description:
{state.get('description_text', '')}

Response format (no explanations, no extra text):
Affected behaviors:
- behavior1
- behavior2
- behavior3

If no behaviors can be identified, respond exactly with:
No affected behaviors identified

Always answer in the same language as the input.
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

        logger.info(f"[Behavior Agent] Output: {result}")

        return {
            "affected_behaviors": result
        }

    except Exception as e:
        logger.error(f"[Behavior Agent] Error: {str(e)}")
        return {
            "affected_behaviors": "No affected behaviors identified",
            "warning": str(e)
        }

