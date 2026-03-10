import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

st.title("RAG AI Assistant")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = OllamaLLM(model="llama3")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

question = st.text_input("Ask a question")

if question:
    result = qa.invoke({"query": question})
    st.write(result["result"])