from typing import List
from langchain import hub
from src.utils.open_router import ChatOpenRouter
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from dotenv import load_dotenv

load_dotenv()

class DialogueAgent:
    def __init__(self, model_name: str = 'openchat/openchat-7b:free') -> None:
        self.model_name = model_name
        self.llm = ChatOpenRouter(model_name=model_name)
        self.prompt = self.create_prompt()
        self.tools = []
        self.runnable_agent = self.create_runnable_agent()
        self.store = {}


    def create_prompt(self):
        return hub.pull("dialogue_agent")
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    
    def create_runnable_agent(self) -> RunnableWithMessageHistory:
        chain = self.prompt | self.llm
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    
    def change_model(self, new_model: str) -> str:
        self.model_name = new_model
        self.llm = ChatOpenRouter(new_model)
        self.runnable_agent = self.create_runnable_agent()
        return f"Modelo cambiado a {new_model}"
    
    def process_user_input(self, user_input: str, session_id: str) -> str:
        if user_input.lower().startswith("cambiar modelo a "):
            new_model = user_input.lower().replace("cambiar modelo a ", "").strip()
            return self.change_model(new_model)
  
        response = self.runnable_agent.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content
    
    def get_model(self) -> str:
        return self.model_name
    
    def get_available_models(self) -> List[str]:
        return [
            'openchat/openchat-7b:free',
            'meta-llama/llama-2-13b-chat'            
        ]
    
    def get_conversation_history(self, session_id: str) -> str:
        return self.runnable_agent.get_session_history(session_id)
    
    def clear_conversation_history(self, session_id: str):
        self.runnable_agent.get_message_history(session_id).clear()
    
    def start_conversation(self):
        print("Asistente: ¡Hola! Soy tu asistente personal.\n"
              f"Estoy usando el modelo {self.get_model()}\n"
              "Puedes cambiar el modelo en cualquier momento diciendo 'cambiar modelo a [nombre del modelo]'.\n"
              "Modelos disponibles:", ", ".join(self.get_available_models()),"\n"
              "¿En qué puedo ayudarte hoy?")
        
        session_id = "1"
        while True:
            user_input = input("Tú: ")
            if user_input.lower() == "exit":
                print("Asistente: ¡Hasta Luego! Que tengas un buen día.")
                break

            dialogue_response = self.process_user_input(user_input, session_id)
            print(f"Asistente: {dialogue_response}")
        
        print("\nHistorial de la conversación:")
        print(self.get_conversation_history(session_id))
