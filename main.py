import os
import openai
import pinecone
import nest_asyncio
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone_api_key = os.environ["PINECONE_KEY"]

pinecone.init(
    api_key="pinecone_api_key",  # find at app.pinecone.io
    environment="asia-southeast1-gcp-free"  # next to api key in console
)


def create_data(url:str, filter_url:list):
    loader = SitemapLoader(
        web_path=url,
        filter_urls= filter_url
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    return chunks, embeddings


def get_creat_index(doc_chunk, embedding, pinecone_name, create_index=False):
    if create_index:
        doc_index = Pinecone.from_documents(doc_chunk, embedding, index_name=pinecone_name)
    else:
        doc_index = Pinecone.from_existing_index(pinecone_name, embedding)

    return doc_index



if __name__ == '__main__':
    main_url = ""

    data_chunk, document_embed = create_data()
