# templates/uwv_agent.py

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder
)

def get_condense_question_prompt():
    template = """
    Gegeven de volgende conversatie en een vervolgvraag, herformuleer de vervolgvraag als een zelfstandige vraag.

    Gespreksgeschiedenis:
    {chat_history}

    Vervolgvraag: {question}

    Zelfstandige vraag:
    """
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )

def get_combine_docs_prompt():
    system_message = SystemMessagePromptTemplate.from_template(
        "Je bent een deskundige en behulpzame assistent van het UWV. "
        "Je beantwoordt vragen over UWV-diensten, uitkeringen, en arbeidsmarktinformatie duidelijk en beknopt. "
        "Omdat het UWV meerdere soorten uitkeringen aanbiedt, moet je altijd controleren of je voldoende informatie hebt voordat je een antwoord geeft. "
        "Vraag altijd naar meer informatie als de vraag onduidelijk is of als het niet duidelijk is over welke uitkering of situatie de cliënt het heeft. "
        "Wees geduldig en zorg ervoor dat je altijd op taalniveau B2 communiceert. "
        "Als je het antwoord niet weet, geef dat eerlijk aan en verwijs de gebruiker naar de officiële UWV-website of klantenservice. "
        "Zorg ervoor dat elk antwoord kort en overzichtelijk is, zodat het in een chatbot-scherm past. Gebruik opsommingstekens (- of •) om informatie overzichtelijk te presenteren. "
        "De few-shot voorbeelden zijn erg belangrijk, neem deze structuur altijd over in je prompt. "
        "URLs zijn altijd dikgedrukt."
    )

    # Voeg de context toe aan de prompt
    context_message = SystemMessagePromptTemplate.from_template(
        "Gebruik de volgende informatie om de vraag te beantwoorden:\n{context}"
    )

    # Few-shot voorbeelden (optioneel)
    example_messages = [
        # Voeg hier eventuele voorbeeldberichten toe als je dat wilt
    ]

    # Opbouw van de volledige prompt
    prompt = ChatPromptTemplate.from_messages(
        [system_message, context_message] +
        example_messages +
        [
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    return prompt
