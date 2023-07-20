import requests
import openai
import os
import gradio as gr
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import (
    PromptTemplate,
)

from langchain.chat_models import ChatOpenAI

openai.api = os.environ["OPENAI_API_KEY"]

if __name__ == '__main__':
    #### create the create vector index store #####
    loader = DirectoryLoader('data/', glob="**/*.txt")
    docs = loader.load()
    embedding = OpenAIEmbeddings()
    len(docs)
    if not os.path.exists("./chroma"):  # create index from the docs
        print("Create index from loader....")
        index = VectorstoreIndexCreator().from_loaders([loader])
    else:
        print("Reuse the created index")
        vectorstore = Chroma(persist_directory="./chroma", embedding_function=embedding)
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)

    # test if the information can be retrieved
    question = "What is the job?"
    index.query(question)

    #### Create prompt template for the bot ####
    prompt_template = """You are a interviewer from TUV Sud, and you are looking for a candidate for the OpenAI expert/Data Scientist Position
        at TUV SUD. Please ask some data science question to the interviewee and also ask some questions which is specifically required for the job.
        {context}
        Interviewer(You): {question}
        Interviewee(The Human):"""
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', "question"])

    #### Chain everything together ####
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8),
        retriever=index.vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    #### Deploy with Gradio ####
    with gr.Blocks() as demo:
        gr.Markdown("## InterviewFreund")
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")
        chat_history = []


        def user(user_message, history):
            print("Type of use msg:", type(user_message))
            chat_history_tuples = []
            for message in chat_history:
                chat_history_tuples.append((message[0], message[1]))
            response = qa_chain({"question": user_message, "chat_history": chat_history_tuples})
            history.append((user_message, response["answer"]))
            print(history)
            return gr.update(value=""), history


        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(debug=True)










