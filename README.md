# 🧠 PDF Question Answering with RAG | Ollama + Groq + Streamlit

This project implements a conversational **Question Answering (QnA)** system over PDF documents using a **Retrieval-Augmented Generation (RAG)** approach. It leverages **Ollama** for local embeddings, **Groq** for LLM inference (e.g. `Gemma2-9b-It`), and provides a simple interactive interface with **Streamlit**.

---

## 📦 Features

- ✅ Upload and process multiple PDFs  
- ✅ Generate local embeddings using Ollama (`mxbai-embed-large`)  
- ✅ Use Groq's hosted LLM (`Gemma2-9b-It`) for chat-style answers  
- ✅ Context-aware query rephrasing using chat history  
- ✅ Session-wise memory for coherent multi-turn conversations  
- ✅ Minimal web UI via Streamlit  

---

## 🛠️ Installation & Setup


### 1. Create and Activate Python Environment

```bash
# With conda
conda create -n ragpdf python=3.10 -y
conda activate ragpdf
```

### 2. Install Required Python Packages

```bash
pip install -r requirements.txt

```

---

## 🤖 Set Up Ollama for Local Embeddings

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start Ollama Server

```bash
ollama serve
```

### 3. Pull the Embedding Model

```bash
ollama pull mxbai-embed-large
```

> This will be used to generate local vector embeddings from PDF content.

---

## 🌐 (Optional) Set Up Groq for LLM

If you're using **Groq's hosted models**, generate a key from: [https://console.groq.com/keys](https://console.groq.com/keys)

Then either:

### a. Set the key in `.env`

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### b. Or Enter it Manually in the Streamlit UI

You will be prompted when the app runs.

---

## 🚀 Run the App

```bash
streamlit run app.py
```

You’ll be able to:

1. Enter your Groq API Key  
2. Upload one or more PDF files  
3. Ask questions about the uploaded content  
4. View concise answers + chat history  

---
