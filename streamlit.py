# chatbot_app.py

import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Laad de OpenAI API-sleutel uit het .env-bestand
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    st.error("OPENAI_API_KEY is niet ingesteld in de omgevingsvariabelen")
    st.stop()

# Initialiseer de OpenAI LLM
llm = ChatOpenAI(
    temperature=0.7,
    openai_api_key=api_key,
    model_name="gpt-4o"  # Gebruik "gpt-3.5-turbo" als je geen toegang hebt tot GPT-4
)

# Laad je FAISS vectorstore
vector_store_path = "vector_store"  # Vervang dit met het pad naar je FAISS indexbestand
if os.path.exists(vector_store_path):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    st.error("Vectorstore niet gevonden. Zorg ervoor dat deze is meegeleverd met de applicatie.")
    st.stop()

# Creëer een retriever van de vectorstore
retriever = vectorstore.as_retriever()

# Initialiseer geheugen met output_key
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

# Importeer je aangepaste prompts uit uwv_agent.py
import uwv_agent

# Haal de prompts op
condense_question_prompt = uwv_agent.get_condense_question_prompt()
combine_docs_prompt = uwv_agent.get_combine_docs_prompt()

# Maak de ConversationalRetrievalChain met je custom prompts
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=st.session_state.memory,
    condense_question_prompt=condense_question_prompt,
    combine_docs_chain_kwargs={'prompt': combine_docs_prompt},
    return_source_documents=True
)

# Streamlit-applicatie
st.title("UWV Chatbot")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("Jij: ", key="input")
    return input_text

user_input = get_text()

if user_input:
    with st.spinner("Beantwoordt..."):
        response = qa_chain({"question": user_input})
        answer = response['answer']
        source_documents = response.get('source_documents', [])
        
        # Haal de relevante stukken en URLs uit de source_documents
        relevant_chunks = [
            {
                'content': doc.page_content,
                'url': doc.metadata.get('url', 'Geen URL beschikbaar')
            } for doc in source_documents
        ]
        
        # Sla het antwoord en de vraag op
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
        
        # Toon het gesprek
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.markdown(f"**Bot:** {st.session_state['generated'][i]}")
            st.markdown(f"**Jij:** {st.session_state['past'][i]}")
        
        # Toon relevante documenten indien aanwezig
        if relevant_chunks:
            st.markdown("### Relevante informatie:")
            for idx, chunk in enumerate(relevant_chunks, 1):
                with st.expander(f"Bron {idx}"):
                    st.markdown(f"**Inhoud:** {chunk['content']}")
                    st.markdown(f"**URL:** {chunk['url']}")
