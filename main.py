from src.agents.rag_agent import RAGAgent


def main():
    print('Inicio proyecto MultiAgent-SecondBrain-AI')

    rag_agent = RAGAgent()
    # rag_agent.load_knowledge_base("/path/to/knowledge_base", ['txt', 'json', 'csv', 'email'])
    # rag_agent.setup_agent()
    
    # result = rag_agent.query("¿Cuál es mi próxima reunión?")
    # print(result)

    # rag_agent.add_document("/path/to/new/document.txt")

    rag_agent.setup_agent('src/data/json-example.json')



if __name__ == "__main__":
    main()