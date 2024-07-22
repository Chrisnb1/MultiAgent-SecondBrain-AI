from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from src.utils.open_router import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

class DialogueAgent:
    def __init__(self, default_model: str = 'meta-llama/llama-2-13b-chat'):
        self.default_model = default_model
        self.current_model = default_model
        self.llm = self.create_llm(default_model)
        self.memory = ConversationBufferMemory()
        self.conversation = self.create_conversation()

    def create_llm(self, model_name: str):
        return ChatOpenRouter(model_name=model_name)
        
    def create_conversation(self):
        return ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def change_model(self, new_model: str):
        self.current_model = new_model
        self.llm = self.create_llm(new_model)
        self.conversation = self.create_conversation()
        return f"Modelo cambiado a {new_model}"
    
    def process_user_input(self, user_input: str) -> str:
        if user_input.lower().startswith("cambiar modelo a "):
            new_model = user_input.lower().replace("cambiar modelo a ", "").strip()
            return self.change_model(new_model)
        
        response = self.conversation.predict(input=user_input)
        return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        return [
            {"role": "human" if isinstance(message, HumanMessage) else "ai", "content": message.content}
            for message in self.memory.chat_memory.messages
        ]
    
    def clear_conversation_history(self):
        self.memory.clear()

    def detect_intent(self, user_input: str) -> str:
        intent_prompt = ChatPromptTemplate.from_template(
            "Detecta la intención principal del siguiente mensaje del usuario. "
            "Responde con una sola palabra: 'chat', 'search', 'help', o 'change_model'.\n\n"
            "Mensaje del usuario: {input}"
        )
        intent_chain = intent_prompt | self.llm
        intent = intent_chain.invoke({"input": user_input}).content.strip().lower()
        return intent if intent in ['chat', 'search', 'help', 'change_model'] else 'chat'

    def get_available_models(self) -> List[str]:
        return [
            'meta-llama/llama-2-13b-chat',
            'openchat/openchat-7b:free'
        ]

    def get_current_model(self) -> str:
        return self.current_model
    
    def start_conversation(self):
        print("Asistente: ¡Hola! Soy tu asistente personal MS-AI.\n"
              f"Estoy usando el modelo {self.get_current_model()}\n"
              "Puedes cambiar el modelo en cualquier momento diciendo 'cambiar modelo a [nombre del modelo]'.\n"
              "Modelos disponibles:", ", ".join(self.get_available_models()),"\n"
              "¿En qué puedo ayudarte hoy?")

        while True:
            user_input = input("Tú: ")
            if user_input.lower() == "exit":
                print("Asistente: ¡Hasta luego! Que tengas un buen día.")
                break

            intent = self.detect_intent(user_input)
            
            if intent == "change_model":
                response = self.process_user_input(user_input)
                print(f"Asistente: {response}\n"
                      f"Modelo actual: {self.get_current_model()}")
            else:
                dialogue_response = self.process_user_input(user_input)
                print(f"Asistente: {dialogue_response}")

        print("\nHistorial de la conversación:")
        for message in self.get_conversation_history():
            print(f"{message['role'].capitalize()}: {message['content']}")
