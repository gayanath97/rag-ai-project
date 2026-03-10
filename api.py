from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

app = FastAPI()

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

# LLM
llm = OllamaLLM(model="llama3")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

@app.get("/ask")
def ask(question: str):

    result = qa.invoke({"query": question})

    return {"answer": result["result"]}