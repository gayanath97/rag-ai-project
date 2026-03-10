## Build a Local RAG App with LangChain, Chroma, and Ollama (Step‑by‑Step Guide)

You don’t need cloud APIs to build a powerful Retrieval‑Augmented Generation (RAG) system. In this tutorial, you’ll create a **fully local RAG app** using:

- **LangChain** for orchestration  
- **Hugging Face** sentence embeddings  
- **Chroma** as a local vector database  
- **Ollama + Llama 3** as the LLM backend  

We’ll cover the **theory** and then walk through the **exact code** you can run. You can paste this entire article and its snippets directly into your Medium post.

---

## 1. What Is RAG?

Large Language Models (LLMs) like Llama 3 are trained on huge datasets, but they:

- Don’t know about your **private documents**
- Can be **outdated** for fast‑moving domains
- Sometimes **hallucinate** answers when they don’t know

**Retrieval‑Augmented Generation (RAG)** fixes this by giving the model access to an **external knowledge base** at query time.

The RAG flow has two main phases.

### 1.1. Index phase (offline)

1. Take your documents.  
2. Split them into smaller chunks.  
3. Convert each chunk into a vector using an **embedding model**.  
4. Store these vectors in a **vector database**.  

### 1.2. Query phase (online)

1. Take the user’s question.  
2. Convert the question into a vector using the **same embedding model**.  
3. Retrieve the most similar chunks from the vector database.  
4. Feed those chunks into the LLM as context and ask it to answer.  

In other words:

> Question → Embedding → Retrieval → Context → LLM Answer

The key idea: the LLM **reads the relevant chunks from your documents** at query time instead of guessing from memory.

---

## 2. Architecture of the Project

We’ll build a very small but complete RAG system with just **two Python scripts**:

- `ingest.py` – build the vector index from a text file  
- `query.py` – ask questions against that index using an LLM  

**Components:**

- **Data**: a text file `data/sample.txt`  
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)  
- **Vector DB**: local Chroma, stored in the `chroma_db/` directory  
- **LLM**: `llama3` served by Ollama on your machine  
- **Orchestration**: LangChain’s `RetrievalQA` chain  

Everything runs **locally**—no OpenAI/Anthropic API keys required.

---

## 3. Project Setup

### 3.1. Directory structure

You can use a simple layout like this:

```text
rag-ai-project/
├─ data/
│  └─ sample.txt
├─ ingest.py
├─ query.py
└─ requirements.txt
```

### 3.2. Python dependencies (`requirements.txt`)

Create a `requirements.txt` file with at least the following:

```text
langchain
langchain-community
langchain-huggingface
langchain-chroma
langchain-ollama
chromadb
sentence-transformers
torch
transformers
numpy
ollama
```

Install them:

```bash
pip install -r requirements.txt
```

### 3.3. Install and run Ollama + Llama 3

1. Install Ollama from `https://ollama.ai` (supports macOS, Linux, Windows).  
2. Pull the Llama 3 model:

```bash
ollama pull llama3
```

3. Make sure the Ollama service is running (for example, `ollama serve` or just launching the app, depending on your OS).

---

## 4. Creating the Knowledge Base (`data/sample.txt`)

This file is your mini “knowledge base.” You can replace this with your own domain later (internal docs, product guides, etc.). For now, here’s a sample describing a fictional SaaS product:

```text
Acme Analytics – Internal Product Overview

Acme Analytics is a web-based platform that helps small and medium businesses understand their sales, marketing, and customer support performance in one place. The platform connects to common data sources such as Shopify, Stripe, HubSpot, and Zendesk, then normalizes the data into a unified analytics model.

Key Product Pillars

1. Sales Analytics
Acme Analytics pulls order and payment data from Shopify and Stripe. It calculates metrics such as total revenue, average order value, customer lifetime value, and churn rate. Users can create custom cohorts based on signup date, campaign source, and geography. The sales dashboard is refreshed every hour by default, with an option to increase to every 15 minutes on the Enterprise plan.

2. Marketing Attribution
The platform tracks UTM parameters and referrer information to attribute revenue to campaigns and channels. It supports first-touch, last-touch, and multi-touch attribution models. Marketing teams can compare channel performance across Google Ads, Meta Ads, and email campaigns. Acme Analytics also integrates with HubSpot to sync campaign performance back into the CRM.

3. Customer Support Insights
By integrating with Zendesk, Acme Analytics provides metrics on ticket volume, first response time, resolution time, and customer satisfaction (CSAT). Managers can view performance by agent, team, and support channel (email, chat, phone). SLA breach alerts can be configured so that if the average first response time exceeds a threshold, managers receive an email notification.

4. Dashboards and Reporting
Users can create custom dashboards by dragging and dropping charts, tables, and KPIs. Dashboards can be shared via secure links or scheduled as PDF email reports. Access controls allow admins to restrict specific dashboards to teams such as Sales, Marketing, or Support. All reports include a “data freshness” indicator so stakeholders know when the numbers were last updated.

Pricing and Plans

Acme Analytics offers three main plans:
- Starter: Includes up to 2 data sources and daily data refresh. Best for early-stage startups.
- Growth: Includes up to 5 data sources, hourly refresh, and advanced attribution models.
- Enterprise: Includes unlimited data sources, 15-minute refresh, SSO (SAML), dedicated support, and custom SLAs.

Security and Compliance

Acme Analytics uses row-level security, encryption in transit (TLS 1.2+) and at rest (AES-256). Role-based access control (RBAC) allows admins to define roles such as Viewer, Analyst, and Admin. The platform is hosted on SOC 2 compliant infrastructure, and audit logs record who accessed or changed key settings. PII fields such as email addresses can be masked for non-admin users.

Onboarding and Support

New customers go through a guided onboarding flow that connects their data sources and suggests recommended dashboards based on industry. The help center includes step-by-step tutorials, video walkthroughs, and troubleshooting guides. Customers on the Enterprise plan receive a dedicated Customer Success Manager who provides quarterly business reviews and best practice recommendations.

Summary

In short, Acme Analytics centralizes revenue, marketing, and support data into a single source of truth. It is designed to help teams answer questions about growth, retention, and operational efficiency without needing a full-time data engineer.
```

Save this under `data/sample.txt`.

---

## 5. Ingestion Script: Building the Vector Index (`ingest.py`)

The ingestion script:

- Loads `sample.txt`  
- Splits it into chunks  
- Embeds those chunks  
- Stores everything in a persistent Chroma database  

Here is the full code:

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the document
loader = TextLoader("data/sample.txt")
documents = loader.load()

# 2. Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # approximate characters per chunk
    chunk_overlap=50  # overlap between chunks for context continuity
)

docs = splitter.split_documents(documents)

# 3. Create the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store vectors in Chroma (persist to disk)
Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="chroma_db",  # folder where Chroma will store its data
)

print("Documents embedded and stored in Chroma successfully.")
```

### 5.1. What’s happening under the hood?

- **TextLoader** converts your raw text file into a LangChain `Document` object.  
- **RecursiveCharacterTextSplitter** divides that document into smaller, overlapping chunks to:
  - Stay within the LLM’s context window  
  - Improve retrieval granularity  
- **HuggingFaceEmbeddings** uses `all-MiniLM-L6-v2` to map each chunk into a dense vector so that similar meanings are close together in vector space.  
- **Chroma.from_documents**:
  - Computes embeddings for all chunks  
  - Stores embeddings, metadata, and original text in a Chroma collection  
  - Persists everything in the `chroma_db/` directory  

### 5.2. Run the ingestion

From the project root:

```bash
python ingest.py
```

You should see a new `chroma_db/` directory appear after this step.

---

## 6. Query Script: Asking Questions via RAG (`query.py`)

Now we’ll build an interactive script that:

- Connects to the same Chroma DB  
- Uses the same embedding model  
- Turns it into a retriever  
- Uses `OllamaLLM` with `llama3`  
- Lets you type questions in a loop  

Here is the full code:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. Load the same embeddings model used during ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load the persisted Chroma vector database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 3. Turn the vectorstore into a retriever
retriever = vectorstore.as_retriever()

# 4. Load the local LLM via Ollama
llm = OllamaLLM(model="llama3")

# 5. Build a RetrievalQA chain: retrieval + generation
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 6. Simple CLI loop
while True:
    question = input("\nEnter your question (or 'exit' to quit): ")
    if question.lower() in {"exit", "quit"}:
        break

    result = qa.invoke({"query": question})

    print("\nAnswer:")
    if isinstance(result, dict) and "result" in result:
        print(result["result"])
    else:
        print(result)
```

### 6.1. How the query pipeline works

1. **Embeddings**  
   We re‑initialize `HuggingFaceEmbeddings` with the same `all-MiniLM-L6-v2` model. This is critical—if you change the model here but not during ingestion, your query vectors won’t be compatible with your stored vectors.

2. **Chroma vector store**  
   `Chroma(persist_directory="chroma_db", embedding_function=embeddings)` opens the exact same vector DB you built with `ingest.py`.

3. **Retriever abstraction**  
   `vectorstore.as_retriever()` gives you an object that:
   - Takes a natural language query  
   - Embeds it  
   - Runs a similarity search against stored vectors  
   - Returns the top‑k most relevant chunks  

4. **LLM via Ollama**  
   `OllamaLLM(model="llama3")` wraps the local Llama 3 model. All inference stays on your machine.

5. **RetrievalQA chain**  
   `RetrievalQA.from_chain_type(...)` composes:
   - The retriever (to fetch context)  
   - The LLM (to generate an answer using that context)  

   It handles:
   - Prompt construction (injecting retrieved text + your question)  
   - Running the LLM  
   - Returning an answer (and optionally the source documents)  

6. **Interactive loop**  
   The `while True` loop lets you ask multiple questions in a single run, and you can type `exit` or `quit` to stop.

### 6.2. Run the query script

Make sure you:

- Have run `python ingest.py` at least once  
- Have Ollama running and `llama3` pulled  

Then run:

```bash
python query.py
```

Try questions like:

- “What is Acme Analytics and who is it designed for?”  
- “What are the differences between the Starter, Growth, and Enterprise plans?”  
- “How does Acme Analytics handle security and compliance?”  
- “What customer support metrics can I see?”  

You should see answers grounded in the content of `sample.txt`.

---

## 7. Theory: Why This Works

### 7.1. Embeddings and semantic search

The embedding model maps text \( t \) to a vector \( E(t) \in \mathbb{R}^d \).

- Semantically similar texts → vectors close together under a metric like cosine similarity.  
- This lets you perform **semantic search** instead of naive keyword search.  

Indexing time:

- For each chunk \( c_i \), compute \( E(c_i) \) and store it in Chroma.  

Query time:

- For a question \( q \), compute \( E(q) \) and retrieve the chunks \( c_i \) whose vectors are closest to \( E(q) \).  

### 7.2. Why not just paste the whole document into the prompt?

- LLMs have a **context window limit**.  
- Long prompts are:
  - Slow  
  - Expensive (token usage)  
  - Noisy (model has to sift through irrelevant parts)  
- Retrieval ensures the model sees only the **most relevant slices** for each question.

### 7.3. Role of the vector database

A vector database like Chroma handles:

- Efficient similarity search (using indexing/ANN techniques)  
- Persistence on disk  
- Storage of metadata and original text  

This makes your RAG system scalable and reusable across sessions.

---

## 8. Ideas for Extensions

Once this minimal RAG app works, you can extend it in several directions:

- **Show sources in answers**  
  Configure `RetrievalQA` to return `source_documents` and print which text chunks were used so users can verify the answer.

- **Ingest multiple files**  
  Use `DirectoryLoader` or loop over all files in the `data/` directory to build a larger knowledge base.

- **Tune retrieval parameters**  
  Use `as_retriever(search_kwargs={"k": 5})` to control how many chunks are retrieved and discuss precision vs recall.

- **Swap embeddings or LLMs**  
  - Try different embedding models (e.g., `all-mpnet-base-v2`) for potentially better semantic understanding.  
  - Switch Ollama models (`mistral`, `phi3`, etc.) and compare answers.  

- **Add a web UI**  
  Wrap the query pipeline in a FastAPI, Flask, or Streamlit app to expose a simple chat interface.

---

## 9. Conclusion

In this article you:

- Learned the **core idea** behind Retrieval‑Augmented Generation.  
- Built a **fully local RAG stack** using LangChain, Chroma, Hugging Face embeddings, and Ollama with Llama 3.  
- Implemented:
  - `ingest.py` to index documents into a vector database.  
  - `query.py` to answer questions by retrieving relevant chunks and passing them to a local LLM.  

From here, you can plug in your own documentation, knowledge bases, and notes to create a custom AI assistant tailored to your data—without leaving your machine or paying for API tokens.

