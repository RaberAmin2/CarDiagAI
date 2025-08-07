from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import json
import logging

# Lade Modelleinstellungen sicher
try:
    with open('agents/bots_settings.json') as f:
        bots = json.load(f)
except Exception as e:
    logging.warning(f"⚠️ Konnte bots_settings.json nicht laden: {e}")
    bots = {}

# Hauptfunktion
def chat_node(state):
    # Sicheres Abrufen mit Fallback auf 'llama3'
    model_name = bots.get("agent_chat", "llama3")

    # LLM initialisieren
    llm = ChatOllama(model=model_name, base_url="http://localhost:11434", temperature=0)

    # Prompt definieren
    prompt = f"""
You are a car diagnostic assistant AI.
Use the following problem description and analysis to answer the user's question briefly and helpfully.

Problem description:
{json.dumps(state.get('description_text', ''), indent=2)}

Identified possible causes:
{json.dumps(state.get('possible_causes', ''), indent=2)}

User question:
{state.get('user_question', '')}

Respond ONLY in the following JSON format:
{{ "chat_response": "Your response here" }}
"""

    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

        # Versuche, JSON zu parsen
        try:
            parsed = json.loads(result)
            response = parsed.get("chat_response", result)
        except json.JSONDecodeError:
            logging.warning("⚠️ LLM-Antwort konnte nicht als JSON interpretiert werden.")
            response = result  # Fallback: rohe Textantwort

        # Chatverlauf aktualisieren
        chat_entry = {
            "question": state.get('user_question', ''),
            "response": response
        }
        chat_history = state.get('chat_history', []) + [chat_entry]

        return {
            "chat_response": response,
            "chat_history": chat_history
        }

    except Exception as e:
        logging.error(f"❌ Fehler im Chat-Agent: {e}")
        return {
            "chat_response": "",
            "warning": str(e)
        }
