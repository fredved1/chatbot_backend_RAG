import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import openai

def get_available_models(api_key):
    openai.api_key = api_key
    try:
        models = openai.Model.list()
        return [model.id for model in models.data if model.id.startswith("gpt")]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []

class LLMMotor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore = None
        self.vector_store_path = "vector_store"  # Pas dit aan naar het juiste pad
        
        self.llm = ChatOpenAI(temperature=0, api_key=api_key)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.qa_chain = None

    def load_vectorstore(self):
        if os.path.exists(self.vector_store_path):
            self.vectorstore = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Vectorstore succesvol geladen.")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory
            )
        else:
            raise FileNotFoundError("Vectorstore niet gevonden. Zorg ervoor dat deze is meegeleverd met de applicatie.")

    def get_response(self, user_input: str):
        if self.qa_chain is None:
            raise ValueError("QA Chain is not initialized. Please load the vectorstore first.")
        
        with get_openai_callback() as cb:
            response = self.qa_chain({"question": user_input})
        
        relevant_docs = self.vectorstore.similarity_search(user_input, k=3)
        
        return {
            'answer': response['answer'],
            'relevant_chunks': [
                {
                    'content': doc.page_content,
                    'url': doc.metadata.get('url', 'Geen URL beschikbaar')
                } for doc in relevant_docs
            ],
            'token_usage': cb.total_tokens
        }

# Functie om de LLMMotor te initialiseren
def initialize_llm_motor():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY is niet ingesteld in de omgevingsvariabelen")
    
    llm_motor = LLMMotor(api_key=api_key)
    llm_motor.load_vectorstore()
    return llm_motor

# Voorbeeld van gebruik
if __name__ == "__main__":
    llm_motor = initialize_llm_motor()
    
    # Voorbeeld van een conversatie
    queries = [
        "Wat zijn de openingstijden van het UWV kantoor?",
        "Kan je me meer vertellen over de diensten die ze aanbieden?",
        "Hoe kan ik een afspraak maken?"
    ]
    
    for query in queries:
        result = llm_motor.get_response(query)
        print(f"Vraag: {query}")
        print(f"Antwoord: {result['answer']}")
        print("Relevante chunks:")
        for i, chunk in enumerate(result['relevant_chunks'], 1):
            print(f"  Chunk {i}:")
            print(f"    Content: {chunk['content'][:100]}...")  # Toon eerste 100 karakters van elke chunk
            print(f"    URL: {chunk['url']}")
        print(f"Token gebruik: {result['token_usage']}")
        print("\n" + "="*50 + "\n")

    # Test get_available_models functie
    print("\nTesting get_available_models function:")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY is niet ingesteld in de omgevingsvariabelen")
    else:
        available_models = get_available_models(api_key)
        if available_models:
            print("Beschikbare GPT modellen:")
            for model in available_models:
                print(f"- {model}")
        else:
            print("Geen GPT modellen gevonden of er is een fout opgetreden bij het ophalen van de modellen.")

__all__ = ['LLMMotor', 'initialize_llm_motor', 'get_available_models']
