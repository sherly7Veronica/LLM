import os
import chainlit as cl
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from chainlit import make_async

from dotenv import load_dotenv
load_dotenv()


# Load OpenAI API key (should be set in your .env or environment)
os.environ.get("OPENAI_API_KEY")

# Define the LLM
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

loader = TextLoader("data/sample.txt", encoding='utf-8')  # Replace with your filename
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
docs = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(docs, embeddings)

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriever)

# Chainlit message handler
@cl.on_message
async def main(message: cl.Message):
    # Get user input from the message
    user_input = message.content

    # Generate a response using the chain
    response = await make_async(qa_chain.invoke)(user_input)

    final_response = response.get("result", "No answer found.") 

    # Send the response back to the user
    await cl.Message(content=final_response).send()
