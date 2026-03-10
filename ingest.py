from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)

print("Loading documents...")

pdf_loader = DirectoryLoader(
    "data/pdfs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

text_loader = DirectoryLoader(
    "data/text",
    glob="*.txt",
    loader_cls=TextLoader
)

documents = pdf_loader.load() + text_loader.load()

print(f"Loaded {len(documents)} documents")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="chroma_db"
)

print("Documents successfully indexed!")