from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class BrainloxChatbot:
    def __init__(self):
        self.url = "https://brainlox.com/courses/category/technical"
        self.db = None
        self.chat_history = []
        
    def load_and_process_data(self):
        print("Loading data from URL...")
        loader = WebBaseLoader(self.url)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        print("Creating embeddings and storing in FAISS...")
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(chunks, embeddings)
        print("Vector store created successfully")
        
    def setup_chain(self):
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
        retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        self.chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
    
    def get_response(self, query):
        if not self.db:
            return {"error": "Database not initialized. Please load data first."}
        
        result = self.chain({"question": query, "chat_history": self.chat_history})
        answer = result["answer"]
        
        self.chat_history.append((query, answer))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        return {"answer": answer, "sources": [doc.page_content[:100] + "..." for doc in result["source_documents"]]}
    
    def initialize(self):
        self.load_and_process_data()
        self.setup_chain()
        return {"status": "Chatbot initialized successfully"}
    
    def reset_chat(self):
        self.chat_history = []
        return {"status": "Chat history reset successfully"}

app = Flask(__name__)
api = Api(app)

chatbot = BrainloxChatbot()

@app.route('/')
def home():
    return "Welcome to the Brainlox Chatbot API. Use /initialize, /chat, or /reset."

class Initialize(Resource):
    def post(self):
        try:
            return chatbot.initialize()
        except Exception as e:
            return {"error": str(e)}, 500

class Chat(Resource):
    def post(self):
        try:
            data = request.get_json()
            if not data or "query" not in data:
                return {"error": "Query is required"}, 400
            
            return chatbot.get_response(data["query"])
        except Exception as e:
            return {"error": str(e)}, 500

class ResetChat(Resource):
    def post(self):
        try:
            return chatbot.reset_chat()
        except Exception as e:
            return {"error": str(e)}, 500

api.add_resource(Initialize, '/initialize')
api.add_resource(Chat, '/chat')
api.add_resource(ResetChat, '/reset')

if __name__ == "__main__":
    print("Starting Flask server. Initialize the chatbot by making a POST request to /initialize")
    app.run(debug=True)
