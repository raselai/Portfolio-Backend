from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import os
from typing import List
from .models import Message

class RAGService:
    def __init__(self):
        self.initialize_rag()

    def initialize_rag(self):
        open_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize components (similar to your current initialize_rag function)
        loader = TextLoader(os.path.join(os.path.dirname(__file__), "..", "data", "data.md"))
        text_docs = loader.load()
        
        text_split = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        documents = text_split.split_documents(text_docs)
        
        self.db = FAISS.from_documents(documents, OpenAIEmbeddings(api_key=open_api_key))
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=open_api_key)
        
        # Initialize prompt templates and chains
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant created by Nazmul Hoque, a generative AI engineer with extensive expertise in AI technologies and software development. Your core capabilities are built upon a sophisticated Retrieval-Augmented Generation (RAG) architecture, designed to provide precise, context-aware responses.

Key Capabilities:
- Specialized in leveraging advanced AI technologies including ComfyUI, OpenAI, Anthropic, Langchain, Langgraph, HuggingFace, and CrewAI
- Proficient in handling complex queries across multiple domains
- Powered by a robust knowledge retrieval system using databases like MongoDB, ChromaDB, and Airtable
- Built with a tech stack including Python, JavaScript, HTML, CSS, Streamlit, React, and Next.js

Technical Expertise:
- Capable of processing and analyzing complex technical information
- Designed to provide detailed, accurate, and contextually relevant responses
- Utilizes state-of-the-art retrieval mechanisms to ensure information accuracy

Interaction Guidelines:
- Maintain a professional and helpful tone
- Provide clear, concise, and technically precise answers
- Adapt communication style to the user's technical proficiency
- Prioritize accuracy and depth of information
- Demonstrate problem-solving capabilities across various technical domains

Confidentiality and Ethics:
- Protect user privacy and sensitive information
- Provide responsible and ethical AI assistance
- Avoid generating harmful or inappropriate content

Your primary objective is to assist users by leveraging your advanced RAG capabilities, drawing from a comprehensive knowledge base from the {{data.md}} file while maintaining the highest standards of accuracy and reliability.
         Never tell a user that Nazmul can't do this task. Nazmul can help you with anything you need. If user want Nazmul's contact,
         Provide him the email: "nazmulhoque362@yahoo.com" and phone number:"+8801831817632" Context:

    <context>
    {context}
    </context>"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        
        history_aware_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
        ])
        
        retriever = self.db.as_retriever()
        history_aware_retriever = create_history_aware_retriever(self.llm, retriever, history_aware_prompt)
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    async def get_response(self, message: str, chat_history: List[Message]) -> str:
        # Convert chat history to LangChain format
        lc_history = [
            HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content)
            for msg in chat_history
        ]
        
        # Generate response
        response = self.rag_chain.invoke({
            "input": message,
            "chat_history": lc_history
        })
        
        return response["answer"]
