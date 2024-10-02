# app.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Import LangChain components
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate

# Import Ollama LLM and Embeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Agent components
from langchain.agents import create_react_agent, Tool, create_tool_calling_agent,AgentExecutor
# from langchain.agents.tool_calling_agent.base import create_tool_calling_agent

# Load environment variables from .env file
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("OLLAMA")

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
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstoredb = FAISS.from_documents(documents, embeddings)
    retriever = vectorstoredb.as_retriever()
    return retriever

# Load documents and create retriever
documents = load_documents()
retriever = create_vectorstore(documents)

# Initialize the Ollama LLM
llm = Ollama(model="mistral")

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
conversation_histories_agent = {}

# Initialize the memory
def initialize_memory(session_id):
    if session_id not in conversation_histories_agent:
        conversation_histories_agent[session_id] = ConversationBufferWindowMemory(k=10, return_messages=True)
        # conversation_histories_agent[session_id] = ConversationBufferMemory(
        #     memory_key="chat_history", return_messages=True
        # )
    return conversation_histories_agent[session_id]

def append_chat_history(memory,input, response):
    memory.save_context({"input": input}, {"output": response})

def invoke(input,memory, agent_executor):
    msg = {
        "input": input,
        "chat_history": memory.load_memory_variables({}),
    }
    print(f"Input: {msg}")

    # response = agent_executor.invoke(msg)
    # print(f"Response: {response}")

    # append_chat_history(memory,response["input"], response["output"])
    # print(f"History: {memory.load_memory_variables({})}")
    # return response["output"]
    try:
        response = agent_executor.invoke(msg)
        print(f"Response: {response}")

        if "output" in response:
            append_chat_history(memory, input, response["output"])
            print(f"History: {memory.load_memory_variables({})}")
            return response["output"]
        else:
            return "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
    except Exception as e:
        print(f"Error: {str(e)}")
        return "I encountered an error while processing your request. Could you please try again?"
    

# Endpoint to receive question and return answer using the agent
@app.post("/agent")
async def agent_endpoint(question: Question):
    # Initialize or get the memory for this session
    memory = initialize_memory(question.session_id)
    
    # Create the agent
    from langchain import hub

    # Get the prompt to use - you can modify this! 
    tool_names = [tool.name for tool in tools]

    from langchain_core.prompts import PromptTemplate

#     template = '''You are an intelligent agent capable of answering questions using the following tools:

# {tools}

# Follow this process for each query:

# Question: The question the user asks
# Thought: Think about what to do next
# Action: Choose one action from the available tools [{tool_names}]
# Action Input: Provide the necessary input for the action
# Observation: Observe and record the result of the action
# ... (This Thought/Action/Action Input/Observation can repeat multiple times as necessary)
# Thought: I now have the final answer
# Final Answer: Provide the final answer to the original question

# Here are some example scenarios for when to use each tool:
# - If the user asks for a joke, use the "Joke Teller" tool.
# - If the query is about specific content from the PDF document, use the "VectorDB" tool to retrieve relevant information.
# - If the input is a general query like "Hello" or "How are you?", respond directly without using any tools.

# Current Question: {input}
# Previous Interactions (Memory): {agent_scratchpad}

# Now proceed!
# '''
    
#     template = '''You are an intelligent agent capable of answering questions using the following tools:

# {tools}

# **Important Instructions:**
# - When specifying an action, **do not** include the word "Use" before the tool name.
# - The `Action` should be **exactly** one of the tool names: {tool_names}.
# - Follow the format precisely.

# Follow this process for each query:

# Question: The question the user asks
# Thought: Think about what to do next
# Action: The action to take, **must be** one of [{tool_names}]
# Action Input: The input to the action
# Observation: The result of the action
# ... (This Thought/Action/Action Input/Observation can repeat multiple times as necessary)
# Thought: I now have the final answer
# Final Answer: Provide the final answer to the original question

# **Example Usage:**

# - **Correct:**
#   - Action: VectorDB
#   - Action Input: "What is the content of section 2.1?"

# - **Incorrect:**
#   - Action: Use VectorDB
#   - Action Input: "What is the content of section 2.1?"

# Here are some example scenarios for when to use each tool:
# - If the user asks for a joke, use the "Joke Teller" tool.
# - If the query is about specific content from the PDF document, use the "VectorDB" tool to retrieve relevant information.
# - If the input is a general query like "Hello" or "How are you?", respond directly without using any tools.

# Current Question: {input}
# Previous Interactions (Memory): {agent_scratchpad}

# Now proceed!
# '''
    template = '''
You are an intelligent agent capable of answering questions using the following tools:

{tools}

Use this format:

Question: The question you must answer
Thought: Consider what to do
Action: The action to take, MUST BE one of {tool_names}
Action Input: The input to the action
Observation: The result of the action

Thought: I now know the final answer
Final Answer: The final answer to the original question

Question: {input}
{agent_scratchpad}
  



'''

    # Create a PromptTemplate instance
    prompt = PromptTemplate.from_template(template)



    # Agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # Return the response
    response = invoke(question.question,  memory, agent_executor)
    return {"answer": response}
