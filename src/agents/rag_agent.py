# src/agents/rag_agent.py

from typing import List
from langchain_community.document_loaders import (
    TextLoader, JSONLoader, CSVLoader, UnstructuredEmailLoader, PythonLoader, UnstructuredMarkdownLoader, PyPDFLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, TokenTextSplitter,
    MarkdownHeaderTextSplitter, PythonCodeTextSplitter
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from src.utils.open_router import ChatOpenRouter 
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()

class RAGAgent:
    def __init__(self):
        self.llm = self.get_llm()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-distilroberta-base-v2")
        self.vector_store = None
        self.agent_executor = None
        self.config = {
            'chunk_size': 1000,
            'chunk_overlap': 200
        }

    def create_vector_store(self, splits):
        return Chroma.from_documents(documents=splits, embedding=self.embeddings)
    
    def create_retriever(self, file_path):
        loader = self.get_loader(file_path)
        documents = loader.load()
        splitter = self.get_splitter(file_path)
        splits = splitter.split_documents(documents)

        self.vector_store = self.create_vector_store(splits)
        
        retriever = self.vector_store.as_retriever()

        return retriever

    def get_retriever_tool(self, retriever):
        retriever_tool = create_retriever_tool(
                retriever,
                "recuperador_base_conocimiento",
                "Busca y devuelve información de la base de conocimiento personal."
            )
        return [retriever_tool]


    def setup_agent(self, file_path: str):

        if file_path is not None:
            retriever = self.create_retriever(file_path)
            tools = self.get_retriever_tool(retriever)
            memory = SqliteSaver.from_conn_string(":memory:")

            self.agent_executor = create_react_agent(self.llm, tools, checkpointer=memory)
            while True:
                question = input("Tú: ")
                self.query(question)
                if question.lower()== "exit":
                    print("Asistente: ¡Hasta Luego! Que tengas un buen día.")
                    break
        else:
            raise ValueError(f"No se pudo leer el archivo: {file_path}")
        
            

    def query(self, question: str) -> str:
        config = {"configurable": {"thread_id": "abc123"}}
        if not self.agent_executor:
            raise ValueError("Agent not set up. Call setup_agent() first.")
        for s in self.agent_executor.stream(
            {"messages": [HumanMessage(content=question)]}, config=config
        ):
            print(s["messages"][-1])
            print("----")

    def get_loader(self, file_path: str):
        if file_path.endswith('.txt'):
            return TextLoader(file_path)
        elif file_path.endswith('.json'):
            return JSONLoader(file_path, jq_schema='.documents[].content')
        elif file_path.endswith('.csv'):
            return CSVLoader(file_path)
        elif file_path.endswith('.pdf'):
            return PyPDFLoader(file_path)
        elif file_path.endswith('.md'):
            return UnstructuredMarkdownLoader(file_path)
        elif file_path.endswith('.py'):
            return PythonLoader(file_path)
        elif file_path.endswith('.eml'):
            return UnstructuredEmailLoader(file_path)


    def get_splitter(self, file_extension: str):
        if file_extension in ['.txt', '.pdf', '.docx', '.doc']:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
        elif file_extension in ['.md', '.markdown']:
            return MarkdownHeaderTextSplitter()
        elif file_extension == '.py':
            return PythonCodeTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
        else:
            return TokenTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
        
    def get_llm(self):
        llm =  ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm





        # documents = loader.load()
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # splits = text_splitter.split_documents(documents)
        # self.vector_store.add_documents(splits)


