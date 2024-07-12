import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class DialogueAgent:
    def __init__(self) -> None:
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def process_input(self, user_input):
        payload = json.dumps({
            "model":"meta-llama/llama-3-8b-instruct:free",
            "messages":[{
                "role":"user",
                "content":user_input
            }]
        })

        try:
            response = requests.post(self.api_url, data=payload, headers=self.headers)
            response.raise_for_status()
            response_data = response.json()
            
            # Extraer solo el contenido de la respuesta
            assistant_response = response_data['choices'][0]['message']['content']
            return assistant_response
        except requests.exceptions.RequestException as error:
            return f"Error al procesar la solicitud: {str(error)}"
        
    def start_conversation(self):
        print("Bienvenido, soy tu asistente personal. ¿En qué puedo ayudarte hoy? (Escribe 'salir' al terminar)")

        while True:
            user_input = input("Tú: ")
            if user_input.lower() == 'salir':
                print("Asistente: Ha sido un placer ayudarte. ¡Hasta luego!")
                break

            response = self.process_input(user_input)
            print(f"Asistente: {response}")
