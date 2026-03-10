from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":10})

# Reranker model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# LLM
llm = OllamaLLM(model="llama3")

while True:

    question = input("\nEnter your question: ")

    # Step 1: retrieve documents
    docs = retriever.get_relevant_documents(question)

    # Step 2: prepare pairs for reranker
    pairs = [(question, doc.page_content) for doc in docs]

    # Step 3: compute scores
    scores = reranker.predict(pairs)

    # Step 4: sort documents by score
    ranked_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    # Step 5: select top 3
    top_docs = [doc for _, doc in ranked_docs[:3]]

    # Step 6: create context
    context = "\n\n".join([doc.page_content for doc in top_docs])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    result = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(result)