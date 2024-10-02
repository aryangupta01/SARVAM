# Loading the libraries

import os
from dotenv import load_dotenv

from langchain_community.llms import ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader

load_dotenv

# Langsmith api keys
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# PDF File Path

file_path = (
    "D:\Langchain\iesc111.pdf"
)



# Text Ingetions
loader = PyPDFLoader(file_path)
loader
docs=loader.load()
docs

# Splitting the text

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)

# Embedding the documents
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="gemma:2b"
)

# storing embedding at vectordb
from langchain_community.vectorstores import FAISS
vectorstoredb=FAISS.from_documents(documents,embeddings)

# calling ollama llm
llm = Ollama("gemma:2b")

## Retrieval Chain, Document chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>


"""
)

document_chain=create_stuff_documents_chain(llm,prompt)
document_chain

retriever=vectorstoredb.as_retriever()
from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)

print(retrieval_chain)

