### RAG AI Assistant – From Theory to Your Project

This document explains **Retrieval-Augmented Generation (RAG)** from first principles and then maps each concept to **your actual code** in this project. It is written so you can use it to **teach juniors** step‑by‑step.

---

### 1. Why RAG? (The Problem It Solves)

- **LLMs have limitations**
  - **Hallucinations**: They may confidently invent facts.
  - **Stale knowledge**: A model is only as up‑to‑date as its training data.
  - **Private data**: Your internal PDFs and notes are not part of the base model.

- **RAG idea in one sentence**
  - **Combine a search engine over your own data with an LLM**, so the model answers **based on retrieved documents** instead of only its internal parameters.

---

### 2. High-Level RAG Architecture

At a high level, a RAG system has **two phases**:

- **(A) Ingestion / Indexing**
  - Collect raw documents (PDFs, text files, etc.).
  - Split them into manageable chunks.
  - Convert chunks into **embeddings** (vectors).
  - Store vectors in a **vector database** (Chroma in your project).

- **(B) Retrieval + Generation (Query Time)**
  - User asks a question.
  - Convert the question into an embedding.
  - Search the vector DB for **similar chunks** (retrieval).
  - Optionally **rerank** them (your `query.py` does this).
  - Feed the best chunks + question into an LLM to generate the final answer.

Your project implements both phases:

- Ingestion: `ingest.py`
- Retrieval + Generation:
  - Simple chain: `app.py` (Streamlit UI) and `api.py` (FastAPI)
  - Advanced with reranking: `query.py` (command-line)

---

### 3. Core Concepts: Embeddings, Vector Stores, Retrievers, Rerankers, LLMs

- **Embeddings**
  - A sentence/paragraph is turned into a **vector of numbers** (e.g. 384‑dim).
  - Similar meanings → vectors close together in vector space.
  - In your code, you use:
    - `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`
  - This means **all text chunks and all queries** are embedded in the same space.

- **Vector Store (Chroma)**
  - Stores pairs: **(embedding vector, original text chunk + metadata)**.
  - Supports **similarity search**: “give me top‑k documents closest to this query vector”.
  - In your code, you use `Chroma` with `persist_directory="chroma_db"` so the index is saved to disk.

- **Retriever**
  - A **wrapper around the vector store** that exposes a nice interface like:
    - `get_relevant_documents(query)`
  - Under the hood: embeds the query → similarity search → returns documents.

- **Reranker (CrossEncoder)**
  - Vector search is fast but approximate.
  - A reranker takes **(query, document)** pairs and scores them more precisely.
  - In your code, you use:
    - `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
  - You first retrieve top‑k (e.g. 10) with Chroma, then rerank and keep the best 3.

- **LLM (Large Language Model)**
  - Generates the final answer as natural language.
  - You use **Ollama** with `llama3`:
    - So your LLM is running locally via the `ollama` backend.

---

### 4. Ingestion Phase – How `ingest.py` Works

File: `ingest.py`

#### 4.1 Loading Documents

You load two types of documents:

- **PDFs** from `data/pdfs` using `PyPDFLoader`
- **Text files** from `data/text` using `TextLoader`

Conceptually:
- **Why?** You want to index any content your juniors might store in PDFs or `.txt` notes.
- **Result:** `documents` is a list where each item is a `Document` containing text + metadata (e.g. file path, page).

#### 4.2 Splitting into Chunks

You use `RecursiveCharacterTextSplitter` with:

- `chunk_size=500`
- `chunk_overlap=50`

Meaning:

- Each long document is broken into chunks of ~500 characters.
- Neighboring chunks overlap by 50 characters to preserve context across boundaries.

Why splitting is important:

- Too long chunks → retrieval becomes less precise.
- Too short chunks → LLM lacks enough context.
- Overlap prevents cutting sentences exactly in half.

#### 4.3 Embedding and Indexing with Chroma

You compute embeddings for each chunk using:

- `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`

Then you build the vector store:

- `Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")`

Key ideas:

- Each chunk becomes:
  - `vector` (embedding)
  - `page_content` (text)
  - `metadata` (source file, page, etc.)
- `persist_directory="chroma_db"` ensures the index is saved between runs.
- After `ingest.py` finishes, your knowledge base is ready.

**Teaching note:**  
You can tell juniors: *“`ingest.py` is a one‑time (or periodic) preprocessing step that turns raw files into a searchable knowledge base.”*

---

### 5. Simple RAG with LangChain – `app.py` and `api.py`

Both `app.py` (Streamlit UI) and `api.py` (FastAPI) follow the **same logical pipeline**, only the front‑end differs.

#### 5.1 Shared Components

Both files:

- **Load embeddings**:
  - `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")`
- **Load existing Chroma DB**:
  - `Chroma(persist_directory="chroma_db", embedding_function=embeddings)`
- **Create a retriever**:
  - `vectorstore.as_retriever(...)`
- **Create an LLM**:
  - `OllamaLLM(model="llama3")`
- **Build a RetrievalQA chain**:
  - `RetrievalQA.from_chain_type(llm=llm, retriever=retriever)`

Conceptually, the `RetrievalQA` chain does:

1. Receive a question.
2. Retrieve relevant chunks via the retriever.
3. Construct a prompt that includes:
   - The retrieved context.
   - The user question.
4. Ask the LLM to answer using that context.
5. Return the final answer.

So, instead of you manually constructing prompts, LangChain hides the boilerplate.

#### 5.2 `app.py` – Streamlit UI

- Sets a title: `"RAG AI Assistant"`.
- Text input box: `st.text_input("Ask a question")`.
- When a question is entered:
  - Calls `qa.invoke({"query": question})`.
  - Displays `result["result"]` as the answer.

**Role:** This is a **simple web UI** where non‑technical users (or juniors) can ask questions via browser.

#### 5.3 `api.py` – FastAPI Endpoint

- Creates a FastAPI app: `app = FastAPI()`.
- Defines endpoint: `GET /ask?question=...`.
- Inside the endpoint:
  - Uses `qa.invoke({"query": question})`.
  - Returns JSON: `{"answer": result["result"]}`.

**Role:** This is a **backend API** that other services or frontends can call programmatically.

Teaching angle:

- `app.py` = **human‑friendly UI**.
- `api.py` = **machine‑friendly API**.
- Both reuse the same retrieval + generation logic.

---

### 6. Advanced RAG with Reranking – `query.py`

File: `query.py`

This script illustrates a **more explicit and flexible RAG pipeline** outside of LangChain’s automatic `RetrievalQA` chain.

#### 6.1 Pipeline Overview

Loop:

1. Read user question from CLI.
2. Retrieve top‑k chunks from Chroma.
3. Rerank these chunks with a CrossEncoder.
4. Select the best top‑n chunks (e.g. 3).
5. Manually construct a prompt with:
   - `Context: ...`
   - `Question: ...`
   - `Answer:`
6. Call the LLM directly with this prompt.

#### 6.2 Retrieval (Top‑k with Chroma)

- `retriever = vectorstore.as_retriever(search_kwargs={"k": 10})`
- `docs = retriever.get_relevant_documents(question)`

So:

- The system takes the user question → embeds it.
- Performs similarity search to get **10 candidate chunks**.

#### 6.3 Reranking with CrossEncoder

- You initialize:
  - `reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
- You create pairs:
  - `pairs = [(question, doc.page_content) for doc in docs]`
- Compute scores:
  - `scores = reranker.predict(pairs)`

Then:

- You zip scores with docs.
- Sort by score in descending order.
- Pick top 3:
  - `top_docs = [doc for _, doc in ranked_docs[:3]]`

**Why rerank?**

- Vector similarity is good but rough.
- CrossEncoder is a **smaller bi‑encoder/transformer** that reads both question and document together and gives a **more accurate relevance score**.
- This usually improves answer quality, especially when your corpus is large.

#### 6.4 Prompt Construction and LLM Call

- Build `context` by joining the top docs’ `page_content`.
- Construct a prompt that explicitly says:
  - “Use the following context to answer the question.”
  - Then context.
  - Then question.
  - Then `Answer:` to cue the model.
- Call `llm.invoke(prompt)`.

Compared to `RetrievalQA`:

- Here, **you** control the exact prompt template.
- This is good for teaching juniors what actually goes into a RAG prompt.

---

### 7. How the Pieces Fit Together (End‑to‑End)

1. **Run ingestion once (or whenever data changes)**  
   - Execute `python ingest.py`.
   - This builds/updates the `chroma_db` folder with your indexed documents.

2. **Run a front‑end or interface**
   - `streamlit run app.py` → Web UI for human interaction.
   - `uvicorn api:app --reload` → FastAPI backend `/ask` endpoint.
   - `python query.py` → CLI with explicit reranking pipeline.

3. **User asks a question**
   - Your system embeds the question.
   - Finds relevant chunks in `chroma_db`.
   - (Optionally) reranks them.
   - LLM answers using that context.

4. **Knowledge update**
   - To add new PDFs or text notes, place them in `data/pdfs` or `data/text`.
   - Re‑run `ingest.py` to update the vector DB.

---

### 8. Theory vs. Your Implementation – Talking Points for Juniors

Here are some **ready‑to‑use teaching points** you can walk your juniors through.

- **Concept: Embedding Space**
  - Theory: Sentences are mapped to points in a high‑dimensional space where semantic similarity becomes numeric distance.
  - In code: `HuggingFaceEmbeddings("all-MiniLM-L6-v2")` in `ingest.py`, `app.py`, `api.py`, `query.py`.

- **Concept: Vector Database**
  - Theory: A specialized database that supports fast nearest‑neighbor search on vectors.
  - In code: `Chroma(..., persist_directory="chroma_db")` and `Chroma.from_documents(...)`.

- **Concept: Retrieval**
  - Theory: Instead of asking an LLM directly, first fetch relevant documents.
  - In code: `vectorstore.as_retriever(...)` and `retriever.get_relevant_documents(question)`.

- **Concept: Reranking**
  - Theory: A more precise model refines the ranking of already‑retrieved candidates.
  - In code: `CrossEncoder(...)` in `query.py` with manual reranking logic.

- **Concept: Augmented Generation**
  - Theory: LLM is conditioned on retrieved context, which grounds its generation.
  - In code:
    - `RetrievalQA.from_chain_type(...)` in `app.py` and `api.py`.
    - Manual prompt in `query.py` built from `top_docs`.

- **Concept: Modularity**
  - Theory: Each stage (ingestion, retrieval, reranking, generation) can be improved independently.
  - In code:
    - Swap embedding models.
    - Change chunk sizes.
    - Change `k` for retrieval or reranking strategy.
    - Swap `llama3` for another Ollama model.

---

### 9. How to Extend This Project (Ideas for Juniors)

You can use these as **mini‑projects** for your juniors:

- **Add metadata‑aware retrieval**
  - Store extra metadata for each chunk (e.g. document type, author, date).
  - Filter retrieval by metadata (e.g. only from PDFs, or only from a given folder).

- **Improve prompt template**
  - Add instructions like:
    - “If the answer isn’t in the context, say ‘I don’t know’.”
    - “Always cite which document the answer came from.”

- **Source highlighting**
  - Show which chunks were used in the answer in the Streamlit UI.

- **Experiment with models**
  - Swap `all-MiniLM-L6-v2` with a larger embedding model and compare quality.
  - Try different Ollama models (e.g. `llama3.1`, `mistral`) and compare responses.

---

### 10. Quick Teaching Summary (A-to-Z)

- **A – Acquire data**: Put PDFs and text into `data/`.
- **B – Break into chunks**: `ingest.py` uses a text splitter.
- **C – Create embeddings**: All chunks → vectors via HuggingFace.
- **D – Deposit in vector DB**: Chroma builds `chroma_db`.
- **E – Embed question**: User question → embedding.
- **F – Find similar chunks**: Use retriever to get candidates.
- **G – Grade (rerank)**: CrossEncoder re‑scores and picks best ones (`query.py`).
- **H – Hand to LLM**: Build a prompt with context + question and call LLM.
- **I – Interpret answer**: Show answer in Streamlit UI, API JSON, or CLI.

If your juniors understand this flow and can point to **where each step lives in your code**, they truly understand RAG **from A to Z** in the context of this project.

