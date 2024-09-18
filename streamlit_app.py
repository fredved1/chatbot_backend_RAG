import streamlit as st
from llm_motor import LLMMotor, get_available_models
import os
from dotenv import load_dotenv

# Laad de .env variabelen in
load_dotenv()

# Haal de API-key op
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configuratie van de sidebar
with st.sidebar:
    if not openai_api_key:
        st.error("API key not found. Please make sure it's set in the .env file.")
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    else:
        st.success("API key loaded successfully!")
        st.write(f"Using API key: {openai_api_key[:5]}...")  # API-key verbergen behalve de eerste 5 karakters
    
    # Toon model selectie en temperatuur slider als API key is ingevoerd
    if openai_api_key:
        models = get_available_models(openai_api_key)
        model = st.selectbox("Select Model", models, index=models.index("gpt-4o") if "gpt-4o" in models else 0)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# Hoofdtitel en beschrijving van de app
st.title("UWV Chatbot")
st.caption("Een Streamlit chatbot aangedreven door OpenAI en LangChain, gespecialiseerd in UWV-diensten")

# Initialiseer LLMMotor en start een nieuwe conversatie als deze nog niet bestaat
if "llm_motor" not in st.session_state:
    st.session_state.llm_motor = LLMMotor(openai_api_key, model, temperature)
    st.session_state.llm_motor.start_new_conversation()

# Toon de gespreksgeschiedenis
for message in st.session_state.llm_motor.get_chat_history():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Verwerk gebruikersinvoer
if prompt := st.chat_input("Stel hier uw vraag over UWV..."):
    if not openai_api_key:
        st.info("Voeg alstublieft uw OpenAI API-sleutel toe om door te gaan.")
        st.stop()

    st.chat_message("user").markdown(prompt)
    
    with st.spinner("Even denken..."):
        response = st.session_state.llm_motor.generate_response(prompt)
    
    st.chat_message("assistant").markdown(response)

# Start nieuwe conversatie knop
if st.button("Start Nieuwe Conversatie"):
    st.session_state.llm_motor.start_new_conversation()
    st.rerun()
