from src.agents.dialogue_agent import DialogueAgent


def main():
    print('Inicio proyecto MultiAgent-SecondBrain-AI')

    agent = DialogueAgent()
    agent.start_conversation()


if __name__ == "__main__":
    main()