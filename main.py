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
    len(docs)
    if not os.path.exists("./chroma_db"):
        print("Create index from document....")
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        db.persist()
    else:
        print("Use the created index")
        # index = VectorstoreIndexCreator().from_loaders([loader])
        # index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "./chroma_db"}).from_loaders([loader])
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    # test if the information can be retrieved
    question = "What is the job?"
    index.query(question)

    template = """
    You are a interviewer from TUV Sud, please ask questions to the interviewee for this data science position.
    Use the following context to ask the question.
    Remember, If you don't what to ask, please just say sorry, I do not knpw, don't try to ask anything.
    {history}
    Human: {human_input}
    Assistant:"""
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    template = "You are a interviewer from tuv sud for the data science position, you are looking for an open AI expert."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # last chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8),
        retriever=index.vectorstore.as_retriever(),
        qa_prompt=prompt
    )

    output =  qa_chain.predict(
        human_input=" You are a interviewer, please ask questions for me about the data science position at TUV Sud."
    )
    print(output)


   # build the interview bot
    with gr.Blocks() as demo:
        gr.Markdown("## TÜV SÜD InterviewFreund ")
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")
        chat_history = []


    def user(user_message, history):
        print("Type of use msg:", type(user_message))
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": history})
        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        print(history)
        return gr.update(value=""), history


    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(debug=True)








