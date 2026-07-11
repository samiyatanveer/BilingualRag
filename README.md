# 🌍 BilingualRAG

**Multilingual Retrieval-Augmented Generation (RAG) Assistant**

A production-ready RAG application that lets you upload documents in any language and ask questions in English, Urdu, Roman Urdu, Hinglish, or 50+ other languages. The system maintains chat history and works seamlessly on free cloud deployment.

**Live Demo:** [https://bilingualrag-nchk2f4fdvzvknjmw3grmy.streamlit.app/](https://bilingualrag-nchk2f4fdvzvknjmw3grmy.streamlit.app/)

---

## 🎯 What This Project Is

This is **a learning artifact**, not a production system. It demonstrates:
- How RAG works end-to-end (retrieval → generation)
- Building multilingual applications
- Deploying ML systems to free tiers
- Persistent storage in cloud environments

The next iteration will swap the QA model for a real LLM (Groq GPT-OSS or similar) to generate **much stronger, more fluent answers** instead of the current extraction-based approach.

---

## 🏗️ Architecture

```
User Document
    ↓
Chunk into paragraphs (400 chars, 50-char overlap)
    ↓
Generate embeddings (Sentence Transformers - multilingual)
    ↓
Store in ChromaDB (local vector database)
    ↓
User asks question
    ↓
Convert question to embedding (same model)
    ↓
Semantic search → retrieve top 3 chunks
    ↓
Pass chunks + question to QA model
    ↓
Generate answer (current: extraction-based)
    ↓
Display answer + sources + chat history
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| **UI** | Streamlit | Fast prototyping, free cloud deployment |
| **Embeddings** | Sentence Transformers (multilingual-MiniLM) | 50+ language support, lightweight |
| **Vector DB** | ChromaDB | Local persistence, no server needed |
| **QA Model** | DeepSet RoBERTa | Fast extraction-based answers |
| **Storage** | Local files + JSON | Survives app restarts, Streamlit Cloud compatible |

---

## ✨ Features

✅ **Multilingual Support** — Upload and ask in any language  
✅ **Persistent Chat History** — Survives app restarts (up to 30 days on free tier)  
✅ **Persistent Vector DB** — Documents indexed permanently  
✅ **Source Attribution** — Shows which document chunks were used  
✅ **Lightweight** — Works on Streamlit Cloud free tier ($0/month)  
✅ **Simple UI** — Claude-style chat interface

---

## 🚀 How to Use

### Online (No Setup)
Visit: [https://bilingualrag-nchk2f4fdvzvknjmw3grmy.streamlit.app/](https://bilingualrag-nchk2f4fdvzvknjmw3grmy.streamlit.app/)

### Local Installation

```bash
git clone https://github.com/samiyatanveer/BilingualRAG
cd BilingualRAG
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`

### Usage Steps

1. **Upload Document**
   - Upload a `.txt` file OR paste text directly
   - Supports any language (English, Roman Urdu, Hinglish, etc.)
   - Click "Load from File" or "Load from Paste"

2. **Ask Questions**
   - Type your question in any language
   - System retrieves relevant document chunks
   - Generates answer based on those chunks

3. **View Sources**
   - Click "Sources" to see which document parts were used
   - Verify the answer's grounding

4. **Chat History**
   - Previous messages persist automatically
   - Clears only when you click "Clear Chat"

---

## 📊 Current Limitations (Why This Is v1)

- **Extraction-only answers** — Copies from document, doesn't synthesize
- **No long-form generation** — Can't write paragraphs, just extracts spans
- **Limited reasoning** — Can't connect ideas across documents
- **Surface-level retrieval** — Semantic search works, but basic compared to dense retrievers

**All these are fixed in v2 by using a real LLM instead of RoBERTa.**

---

## 🔮 Next Version (v2) — The Real Thing

The next iteration will use **Groq GPT-OSS** (or similar LLM) to:

✨ **Generate coherent answers** — Synthesize from multiple sources, not just extract  
✨ **Handle complex questions** — "Compare X and Y across documents"  
✨ **Multi-turn reasoning** — Build on previous answers  
✨ **Better language understanding** — True multilingual generation, not just search  

**Architecture change:**
```
Current: Chunk → Embed → Search → Extract
Next:    Chunk → Embed → Search → LLM(retrieved_chunks + question) → Generate
```

This makes answers **dramatically better** — from "here's a sentence from your doc" to "here's a thoughtful answer built from your doc."

---

## 📚 Code Structure

```
app.py
├── Session state initialization (messages, DB, documents)
├── Model loading (cached: embeddings, QA)
├── ChromaDB initialization (persistent storage)
├── Helper functions:
│   ├── chunk_document() — split text into overlapping chunks
│   ├── add_documents_to_db() — embed & store
│   ├── retrieve_relevant_chunks() — semantic search
│   ├── answer_question() — extract answer from chunks
│   ├── load/save_chat_history() — persistence
├── UI (Streamlit):
│   ├── Document upload interface
│   ├── Chat interface (Claude-style)
│   ├── Message history display
│   └── Source attribution
```

**No complex patterns** — straightforward, readable code meant for learning.

---

## 🎓 What You Learn From This

1. **RAG Basics** — How retrieval + generation work together
2. **Embeddings** — Converting text to numbers for semantic search
3. **Vector Databases** — Storing and searching embeddings
4. **Multilingual NLP** — One model for 50+ languages
5. **Streamlit Deployment** — From local to production in minutes
6. **Persistent Storage** — Chat history & vector DB survive restarts

## 🔧 Requirements

```
streamlit==1.28.0
sentence-transformers==2.2.2
transformers==4.35.0
chromadb==0.4.10
langdetect==1.0.9
```

All lightweight and free-tier compatible.

---

## 📖 How to Extend This

**Easy wins for v2:**
1. Replace `qa_model` with Groq LLM API
2. Add document upload as PDF/DOCX (not just `.txt`)
3. Add chat export (download conversation as markdown)
4. Add document management (delete/replace indexed docs)
5. Add reranking (semantic search → reranker → LLM)

**Medium difficulty:**
6. Fine-tune embeddings on domain-specific text
7. Add keyword filtering before semantic search
8. Implement streaming responses (token-by-token)

---

## 🌟 Lessons Learned

- **Streamlit is fast** — prototype to deployment in one session
- **Semantic search is powerful** — retrieves surprisingly relevant chunks without training
- **Persistent storage matters** — chat history changes UX from "demo" to "tool"
- **Free tiers work** — ChromaDB (local) + Streamlit Cloud (free) = $0 deployment
- **Multilingual is free** — one model, 50+ languages, no extra cost

---

## 📝 License

Open source. Use, modify, learn.

---

## 🙏 Credits

- **Sentence Transformers** — Multilingual embeddings
- **ChromaDB** — Vector database
- **Streamlit** — UI & deployment
- **HuggingFace** — Models & transformers library
- **Groq** — Upcoming LLM integration (v2)

---

## 🎯 My Learning Arc

This project is part of my progression toward becoming an Agentic AI Engineer:

1. ✅ **Mini-GPT** — Built transformer architecture from scratch
2. ✅ **Sentiment Classifier** — Transfer learning + production deployment
3. ✅ **BilingualRAG** — Full RAG pipeline
4. 🚀 **Next** — Agentic AI with tool use, planning, and reasoning

---

**Questions? Issues? Feedback?**  
Open an issue or reach out via LinkedIn: [linkedin.com/in/samiya-tanveer](https://linkedin.com/in/samiya-tanveer)

**GitHub:** [github.com/samiyatanveer](https://github.com/samiyatanveer)
