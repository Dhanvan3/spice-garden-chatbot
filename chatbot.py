from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()

def load_knowledge_base():
    loader = TextLoader("restaurant_data.txt")
    documents = loader.load()
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    
    return vectorstore

def create_chatbot(vectorstore):
    #llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    chatbot = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return chatbot

def ask(chatbot, question):
    response = chatbot.invoke({"query": question})
    return response["result"]

if __name__ == "__main__":
    print("Loading Spice Garden knowledge base...")
    vectorstore = load_knowledge_base()
    chatbot = create_chatbot(vectorstore)
    print("Ready! Ask me anything about Spice Garden.")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = ask(chatbot, question)
        print(f"Bot: {answer}\n")
