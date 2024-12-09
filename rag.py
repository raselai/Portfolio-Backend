import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to initialize the RAG components
@st.cache_resource
def initialize_rag():
    open_api_key = os.getenv("OPENAI_API_KEY")
    
    loader = TextLoader("data.md")
    text_docs = loader.load()
    
    text_split = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    documents = text_split.split_documents(text_docs)
    
    db = FAISS.from_documents(documents, OpenAIEmbeddings(api_key=open_api_key))
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=open_api_key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """ You are an AI assistant created by Nazmul Hoque, a generative AI engineer with extensive expertise in AI technologies and software development. Your core capabilities are built upon a sophisticated Retrieval-Augmented Generation (RAG) architecture, designed to provide precise, context-aware responses.

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
    </context>
             """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    history_aware_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    
    retriever = db.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_prompt)
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Streamlit UI
st.title("Laptop Store Assistant")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize RAG components
rag_chain = initialize_rag()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask about laptops...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Convert chat history to the format expected by the RAG chain
        rag_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in st.session_state.chat_history[:-1]  # Exclude the latest user message
        ]
        
        # Generate response
        ai_response = rag_chain.invoke({"input": user_input, "chat_history": rag_history})
        full_response = ai_response["answer"]
        
        message_placeholder.markdown(full_response)
    
    # Add AI response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})