# streamlit.py

import streamlit as st
from llm_motor import LLMMotor, get_available_models
import os
from dotenv import load_dotenv
import openai  # Gebruik de openai-module direct

# Laad de .env variabelen in
load_dotenv()

# Haal de API-sleutel op
openai_api_key = os.getenv("OPENAI_API_KEY")

# Sidebar configuratie
with st.sidebar:
    if not openai_api_key:
        st.error("API-sleutel niet gevonden. Zorg ervoor dat deze is ingesteld in het .env-bestand.")
        openai_api_key = st.text_input("OpenAI API-sleutel", key="chatbot_api_key", type="password")
    else:
        st.success("API-sleutel succesvol geladen!")
        st.write(f"Gebruikte API-sleutel: {openai_api_key[:5]}...")  # Verberg een deel van de API-sleutel

    # Toon modelselectie en temperatuurinstelling als de API-sleutel is ingevoerd
    if openai_api_key:
        models = get_available_models(openai_api_key)
        default_model = "gpt-4" if "gpt-4" in models else models[0]
        model = st.selectbox("Selecteer Model", models, index=models.index(default_model))
        temperature = st.slider("Temperatuur", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    else:
        model = "gpt-3.5-turbo"
        temperature = 0.7

# Hoofdtitel en beschrijving van de app
st.title("UWV Chatbot")
st.caption("Een Streamlit-chatbot aangedreven door OpenAI en LangChain, gespecialiseerd in UWV-diensten.")

# Initialiseer LLMMotor en start een nieuwe conversatie als deze nog niet bestaat
if "llm_motor" not in st.session_state:
    if not openai_api_key:
        st.warning("Voer je OpenAI API-sleutel in om te beginnen.")
        st.stop()
    st.session_state.llm_motor = LLMMotor(openai_api_key, model, temperature)
    opening_message = st.session_state.llm_motor.start_new_conversation()
    st.session_state.messages = [{"role": "assistant", "content": opening_message}]

# Toon de gespreksgeschiedenis
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Verwerk gebruikersinvoer
if prompt := st.chat_input("Stel hier uw vraag over UWV..."):
    if not openai_api_key:
        st.info("Voeg alstublieft uw OpenAI API-sleutel toe om door te gaan.")
        st.stop()

    # Voeg het gebruikersbericht toe aan de weergave
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Even denken..."):
        # Genereer een reactie met behulp van LLMMotor
        response = st.session_state.llm_motor.generate_response(prompt)
        answer = response["answer"]  # Haal het antwoord op
        source_documents = response["source_documents"]  # Haal de bronnen op

    # Voeg het antwoord van de assistent toe aan de weergave
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Toon maximaal 3 relevante documenten
    if source_documents:
        relevant_chunks = [
            {
                'content': doc.page_content,
                'url': doc.metadata.get('url', 'Geen URL beschikbaar')
            } for doc in source_documents[:3]  # Beperk tot 3 bronnen
        ]
        st.markdown("### Relevante informatie:")
        for idx, chunk in enumerate(relevant_chunks, 1):
            with st.expander(f"Bron {idx}"):
                st.markdown(f"**Inhoud:** {chunk['content']}")
                st.markdown(f"**URL:** {chunk['url']}")

    # Verplaats de 4e uitklapper buiten de if-blok
    with st.expander("Gespreksgeschiedenis"):
        chat_history = st.session_state.llm_motor.get_chat_history()
        for msg in chat_history:
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

        # Knop om het geheugen te legen
        if st.button("Geheugen Wissen"):
            st.session_state.llm_motor.clear_memory()
            st.session_state.messages = []
            st.success("Gespreksgeschiedenis is gewist.")
            st.experimental_rerun()

# Start nieuwe conversatie knop
if st.button("Start Nieuwe Conversatie"):
    st.session_state.llm_motor.start_new_conversation()
    st.session_state.messages = [{"role": "assistant", "content": st.session_state.llm_motor.memory.chat_memory.messages[-1].content}]
    st.experimental_rerun()
