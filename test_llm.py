from llm_motor import create_chat_model, generate_response
from dotenv import load_dotenv
import os

# Laad de .env variabelen
load_dotenv()

# Haal de API-key op
openai_api_key = os.getenv("OPENAI_API_KEY")

def test_llm():
    # Maak een chat model
    chat_model = create_chat_model(openai_api_key, model="gpt-4o", temperature=0.7)

    # Simuleer een gesprek
    messages = [
        {"role": "assistant", "content": "Hoe kan ik u vandaag helpen met UWV-gerelateerde vragen?"},
        {"role": "user", "content": "Wat zijn de voorwaarden voor een WW-uitkering?"}
    ]

    # Genereer een antwoord
    response = generate_response(chat_model, messages)

    print("Gebruiker: Wat zijn de voorwaarden voor een WW-uitkering?")
    print(f"Assistent: {response}")

    # Test een vervolgvraag
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "En hoe lang duurt een WW-uitkering maximaal?"})

    response = generate_response(chat_model, messages)

    print("\nGebruiker: En hoe lang duurt een WW-uitkering maximaal?")
    print(f"Assistent: {response}")

if __name__ == "__main__":
    test_llm()