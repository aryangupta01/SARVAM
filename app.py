# # app.py

# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from pydantic import BaseModel

# # Import LangChain components
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_core.prompts import ChatPromptTemplate

# # Import Ollama LLM and Embeddings
# from langchain_community.llms import Ollama
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.document_loaders import PyPDFLoader

# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Load environment variables from .env file
# load_dotenv()

# # Initialize FastAPI app
# app = FastAPI()

# # Pydantic model for request body
# class Question(BaseModel):
#     question: str
#     session_id: str

# # In-memory storage for conversation history per session
# conversation_histories = {}

# # Load and process the PDF document
# def load_documents():
#     file_path = r"D:\Langchain\iesc111.pdf"  # Update with your PDF file path
#     loader = PyPDFLoader(file_path)
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = text_splitter.split_documents(docs)
#     return documents

# # Create vector store and retriever
# def create_vectorstore(documents):
#     embeddings = OllamaEmbeddings(model="llama3.1:8b")
#     vectorstoredb = FAISS.from_documents(documents, embeddings)
#     retriever = vectorstoredb.as_retriever()
#     return retriever

# # Load documents and create retriever
# documents = load_documents()
# retriever = create_vectorstore(documents)

# # Initialize the Ollama LLM
# llm = Ollama(model="llama3.1:8b")

# # Define the prompt template
# prompt_template = ChatPromptTemplate.from_template(
#     """
# Answer the following question based only on the provided context:
# <context>
# {context}
# </context>

# Question: {question}

# Answer:
# """
# )

# # Endpoint to receive question and return answer
# @app.post("/ask")
# async def ask_question(question: Question):
#     # Get or create conversation history for the session
#     session_id = question.session_id
#     if session_id not in conversation_histories:
#         conversation_histories[session_id] = ConversationBufferMemory(
#             memory_key="chat_history", return_messages=True
#         )
#     memory = conversation_histories[session_id]

#     # Create a ConversationalRetrievalChain with memory
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         combine_docs_chain_kwargs={'prompt': prompt_template}
#     )

#     # Run the chain with the user's question
#     response = conversation_chain({"question": question.question})

#     # Return the answer
#     return {"answer": response['answer']}


# app.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Import LangChain components
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# Import Ollama LLM and Embeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Import Agent components
from langchain.agents import Tool, initialize_agent, AgentType

# Load environment variables from .env file
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("OLLAMA")

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for request body
class Question(BaseModel):
    question: str
    session_id: str

# In-memory storage for conversation history per session
conversation_histories = {}

# Load and process the PDF document
def load_documents():
    file_path = r"D:\Langchain\iesc111.pdf"  # Update with your PDF file path
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    return documents

# Create vector store and retriever
def create_vectorstore(documents):
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    vectorstoredb = FAISS.from_documents(documents, embeddings)
    retriever = vectorstoredb.as_retriever()
    return retriever

# Load documents and create retriever
documents = load_documents()
retriever = create_vectorstore(documents)

# Initialize the Ollama LLM
llm = Ollama(model="llama3.1:8b")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {question}

Answer:
"""
)

# Endpoint to receive question and return answer
@app.post("/ask")
async def ask_question(question: Question):
    # Get or create conversation history for the session
    session_id = question.session_id
    if session_id not in conversation_histories:
        conversation_histories[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    memory = conversation_histories[session_id]

    # Create a ConversationalRetrievalChain with memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt_template}
    )

    # Run the chain with the user's question using 'invoke' method
    response = conversation_chain.invoke({"question": question.question})

    # Return the answer
    return {"answer": response['answer']}

# Define additional tools for the agent
def vector_db_query(query: str) -> str:
    # Use the retriever to get relevant documents
    docs = retriever.get_relevant_documents(query)
    # Concatenate the contents of the documents
    contents = "\n".join([doc.page_content for doc in docs])
    return contents

def tell_joke(query: str) -> str:
    joke_prompt = "Tell me a joke."
    joke = llm(joke_prompt).strip()
    return joke


# Define the tools
tools = [
    Tool(
        name="VectorDB",
        func=vector_db_query,
        description="Use this tool to answer questions about the PDF document."
    ),
    Tool(
        name="Joke Teller",
        func=tell_joke,
        description="Use this tool when the user wants to hear a joke."
    ),
]

# Endpoint to receive question and return answer using the agent
@app.post("/agent")
async def agent_endpoint(question: Question):
    # Get or create conversation history for the session
    session_id = question.session_id
    if session_id not in conversation_histories:
        conversation_histories[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    memory = conversation_histories[session_id]

    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

    # Run the agent with the user's question
    response = agent.run(question.question)

    # Return the answer
    return {"answer": response}
