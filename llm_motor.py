from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from typing import List, Dict

# Importeer je aangepaste prompts uit uwv_agent.py
from uwv_agent import get_condense_question_prompt, get_combine_docs_prompt

def create_chat_model(api_key: str, model: str = "gpt-4", temperature: float = 0.7):
    return ChatOpenAI(
        openai_api_key=api_key,
        model=model,
        temperature=temperature
    )

class LLMMotor:
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7):
        self.chat_model = create_chat_model(api_key, model, temperature)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Laad je FAISS vectorstore
        vector_store_path = "vector_store"  # Vervang dit met het juiste pad
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()
        
        # Haal de prompts op
        condense_question_prompt = get_condense_question_prompt()
        combine_docs_prompt = get_combine_docs_prompt()
        
        # Maak de ConversationalRetrievalChain met je custom prompts
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=condense_question_prompt,
            combine_docs_chain_kwargs={'prompt': combine_docs_prompt},
            return_source_documents=True
        )

    def generate_response(self, prompt: str) -> str:
        response = self.qa_chain({"question": prompt})
        answer = response['answer']
        return answer

    def get_chat_history(self) -> List[Dict[str, str]]:
        return [
            {"role": "user" if msg.type == "human" else "assistant", "content": msg.content}
            for msg in self.memory.chat_memory.messages
        ]

    def clear_memory(self):
        self.memory.clear()

    def start_new_conversation(self) -> str:
        self.clear_memory()
        opening_message = "Hallo! Hoe kan ik u vandaag helpen met UWV-gerelateerde vragen?"
        self.memory.chat_memory.add_ai_message(opening_message)
        return opening_message

def get_available_models(api_key: str) -> List[str]:
    import openai
    openai.api_key = api_key
    models = openai.Model.list()
    return [model.id for model in models['data'] if model.id.startswith("gpt")]
