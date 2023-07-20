import requests
import openai
import os
import gradio as gr
from bs4 import BeautifulSoup
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI

openai.api = os.environ["OPENAI_API_KEY"]



if __name__ == '__main__':

    loader = DirectoryLoader('data/', glob="**/*.txt")
    docs = loader.load()
    embedding = OpenAIEmbeddings()
    len(docs)
    if not os.path.exists("./chroma_db"): # create index from the docs
        print("Create index from loader....")
        index = VectorstoreIndexCreator().from_loaders([loader])
    else:
        print("Reuse the created index")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    # test if the information can be retrieved
    question = "What is the job?"
    index.query(question)

    prompt_template = """You are a interviewer from TUV Sud, and your candidate is interviewing for the OpenAI expert/Data Scientist Position
    Please ask some data science question to the interviewee and also ask some questions which is specifically required for the job.
    {context}
    Interviewer: {question}
    Interviewee:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=['context',"question"])

    # last chain, chat over doc with history
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8),
        retriever=index.vectorstore.as_retriever(),
        combine_docs_chain_kwargs = {"prompt": prompt}
    )








