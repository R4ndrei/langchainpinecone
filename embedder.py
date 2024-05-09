
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

## Lets Read the document
def read_doc(data):
    file_loader=PyPDFDirectoryLoader(data)
    documents=file_loader.load()
    return documents

doc=read_doc('data/')
len(doc)

## Divide the docs into chunks
def chunk_data(docs, chunk_size=500, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

documents=chunk_data(docs=doc)
print(len(documents))

## Embedding Technique Of OPENAI
embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
print(embeddings)

vectors=embeddings.embed_query("How are you?")
print(len(vectors))