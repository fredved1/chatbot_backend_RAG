from llm_motor import LLMMotor
from dotenv import load_dotenv
import os

# Laad de .env variabelen
load_dotenv()

# Haal de API-key op
openai_api_key = os.getenv("OPENAI_API_KEY")

def test_llm():
    # Initialiseer de LLMMotor met memory
    llm_motor = LLMMotor(openai_api_key, model="gpt-4", temperature=0.7)

    # Voer een reeks vragen uit en toon de antwoorden
    questions = [
        "Ik ben Thomas",
        "Woon in Amsterdam",
        "Ik houd van pizza",
        "Wat hebben we besproken"
    ]

    for question in questions:
        print(f"Gebruiker: {question}")
        response = llm_motor.generate_response(question)
        print(f"Assistent: {response}\n")

    # Geef de volledige conversatiegeschiedenis terug
    return llm_motor.get_chat_history()

if __name__ == "__main__":
    # Voer de test_llm functie uit en sla het resultaat op
    conversation = test_llm()
    
    # Print de volledige conversatie
    print("\nVolledige conversatie:")
    for message in conversation:
        print(f"{message['role'].capitalize()}: {message['content']}")
