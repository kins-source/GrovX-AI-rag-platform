# GROX AI : Hybrid Agentic QA System for SQL & Document Retrieval,vector search using Agentic RAG

An intelligent AI-powered assistant that enables users to query both structured databases and unstructured documents using natural language.

---

## 📌 Overview

This project combines **Agentic AI** and **Retrieval-Augmented Generation (RAG)** to build an enterprise-level knowledge system that:

* Understands user queries in natural language
* Automatically decides whether to query a database or retrieve document content
* Generates accurate and context-aware responses
* Provides source attribution for transparency

---

## 🧠 Key Features

* 💬 Natural Language Query Interface
* 🤖 Agentic AI (LangGraph-based decision making)
* 🗄️ SQL Query Automation (SQLite)
* 📄 Document Retrieval using RAG (ChromaDB)
* 🔍 Source Attribution (like Perplexity AI)
* ⚡ Fast and Interactive UI (Streamlit)

---

## 🏗️ System Architecture

User → Frontend → Backend → Agent (LLM) ↔ Tools → Response + Sources

* Frontend: Streamlit
* Backend: FastAPI
* Agent: LangGraph + LangChain
* LLM: Ollama (LLaMA 3.1)
* Database: SQLite
* Vector DB: ChromaDB

---

## ⚙️ Tech Stack

Frontend: Streamlit
Backend: FastAPI
LLM: Ollama (LLaMA 3.1)
Agent Framework: LangGraph + LangChain
Vector Database: ChromaDB
Embeddings: all-MiniLM-L6-v2
Structured DB: SQLite

---

## 🔄 Working Flow

1. User enters query in natural language
2. Backend sends query to Agent
3. Agent decides:

   * SQL Tool (for structured queries)
   * Retriever Tool (for documents)
4. Relevant data is retrieved
5. LLM generates response
6. Sources are displayed along with answer

---

## 🧪 Sample Queries

* What is total revenue?
* How many transactions are there?
* Summarize uploaded document
* What is the main topic of this file?

---

## ⚠️ Challenges & Solutions

* Context leakage → Solved using metadata filtering
* Wrong tool selection → Improved system prompts
* Static SQL responses → Prompt-engineered tool docstrings
* UI visibility issues → Fixed using CSS styling

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/kins-source/GrovX-AI-enterprise-rag-platform.git
cd GrovX-AI-enterprise-rag-platform
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Run LLM (Ollama)

Make sure Ollama is installed.

### Start Ollama

```bash
ollama serve
```

### Pull Model (first time only)

```bash
ollama pull llama3.1
```

### Verify Model

```bash
ollama run llama3.1
```

---

### 3. Run Backend

```bash
python -m uvicorn backend.main:app --reload
```

---

### 4. Run Frontend

```bash
python3 -m streamlit run frontend/app.py
```

---

## 🔐 Security Note

Sensitive data such as API keys are stored in `.env` and excluded using `.gitignore`.

---

## 📈 Future Scope

* Multi-document querying
* Cloud deployment
* Multi-user authentication
* Real-time data integration
* Advanced analytics dashboard

---



