import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import json
from pathlib import Path
import os
from datetime import datetime
from langdetect import detect, LangDetectException

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .message-user { background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .message-assistant { background-color: #f0f4c3; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .header-main { font-size: 2.5rem; font-weight: bold; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
""", unsafe_allow_html=True)

CHROMA_DIR = Path("./chroma_data")
CHROMA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = Path("./chat_history.json")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
if "db" not in st.session_state:
    st.session_state.db = None
if "collection" not in st.session_state:
    st.session_state.collection = None

@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

embeddings_model = load_embeddings_model()
qa_model = load_qa_model()

def init_chroma_db():
    if st.session_state.db is None:

        # Create persistent Chroma database
        st.session_state.db = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        try:
            st.session_state.collection = st.session_state.db.get_collection(
                name="documents"
            )
            st.session_state.documents_loaded = True

        except Exception:
            st.session_state.collection = st.session_state.db.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )

def load_chat_history():
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            st.session_state.messages = json.load(f)

def save_chat_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def chunk_document(text, chunk_size=400, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def add_documents_to_db(documents_text):
    chunks = chunk_document(documents_text)

    embeddings = embeddings_model.encode(
        chunks,
        convert_to_numpy=True
    ).tolist()

    st.session_state.collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=[f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
    )

    st.session_state.documents_loaded = True
    return len(chunks)

def retrieve_relevant_chunks(question, top_k=3):
    if not st.session_state.documents_loaded:
        return []
    
    try:
        results = st.session_state.collection.query(
            query_texts=[question],
            n_results=top_k
        )
        return results['documents'][0] if results['documents'] else []
    except:
        return []

def answer_question(question, context):
    if not context:
        return "No relevant documents found. Please upload documents first."
    
    full_context = " ".join(context)
    
    try:
        result = qa_model(question=question, context=full_context)
        return result['answer']
    except Exception as e:
        return f"Could not generate answer. Try asking differently."

init_chroma_db()
load_chat_history()

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="header-main">📚 Document Q&A</p>', unsafe_allow_html=True)
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        save_chat_history()
        st.rerun()

st.markdown("Ask questions about your documents in **any language** — English, Urdu, Roman Urdu, Hinglish, etc.")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "📊 Info"])

with tab2:
    st.subheader("Upload Your Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Option 1: Upload File")
        uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
        
        if uploaded_file:
            if st.button("📥 Load from File", use_container_width=True, type="primary"):
                text_content = uploaded_file.read().decode('utf-8', errors='ignore')
                with st.spinner("Processing..."):
                    num_chunks = add_documents_to_db(text_content)
                st.success(f"✅ Loaded {num_chunks} chunks")
    
    with col2:
        st.markdown("#### Option 2: Paste Text")
        document_text = st.text_area(
            "Paste document text",
            height=200,
            label_visibility="collapsed",
            placeholder="Your text here...\n\nآپ رومن اردو میں بھی پیسٹ کر سکتے ہیں"
        )
        
        if document_text:
            if st.button("📥 Load from Paste", use_container_width=True, type="primary"):
                with st.spinner("Processing..."):
                    num_chunks = add_documents_to_db(document_text)
                st.success(f"✅ Loaded {num_chunks} chunks")
    
    st.divider()
    
    if st.session_state.documents_loaded:
        st.info(f"✅ **Documents indexed:** {st.session_state.collection.count()} chunks stored")
    else:
        st.warning("⚠️ No documents loaded yet. Upload or paste text above.")

with tab1:
    st.subheader("Ask Questions")
    
    if not st.session_state.documents_loaded:
        st.warning("📄 Please upload documents first in the **Documents** tab")
    else:
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_input(
                "Your question:",
                placeholder="What is... / یہ کیا ہے... / Iska matlab kya hai...",
                label_visibility="collapsed"
            )
        
        with col2:
            st.write("")
            ask_btn = st.button("🔍 Ask", use_container_width=True, type="primary")
        
        if ask_btn and question:
            st.session_state.messages.append({"role": "user", "content": question, "time": datetime.now().isoformat()})
            
            with st.spinner("🔍 Searching & generating answer..."):
                chunks = retrieve_relevant_chunks(question)
                answer = answer_question(question, chunks)
            
            st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks, "time": datetime.now().isoformat()})
            save_chat_history()
            st.rerun()
    
    st.divider()
    st.subheader("💬 Conversation")
    
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                st.markdown(f'<div class="message-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-assistant"><b>Assistant:</b> {msg["content"]}</div>', unsafe_allow_html=True)
                
                if "chunks" in msg and msg["chunks"]:
                    with st.expander("📍 Show sources"):
                        for j, chunk in enumerate(msg["chunks"], 1):
                            st.caption(f"**Source {j}:**")
                            st.text(chunk[:150] + "..." if len(chunk) > 150 else chunk)
    else:
        st.info("💬 No messages yet. Ask a question to start!")

with tab3:
    st.subheader("📊 System Info")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.documents_loaded:
            st.metric("Chunks Indexed", st.session_state.collection.count())
        else:
            st.metric("Chunks Indexed", "0")
    
    with col2:
        st.metric("Chat Messages", len(st.session_state.messages))
    
    with col3:
        st.metric("Model", "Multilingual")
    
    st.divider()
    st.markdown("""
    #### 🌍 Supported Languages
    English, Urdu, Roman Urdu, Hinglish, Arabic, Spanish, French, German, Chinese, Japanese, and 50+ more languages.
    
    #### 🔧 How It Works
    1. Upload your documents (any language)
    2. System creates embeddings (numerical representations)
    3. You ask questions in any language
    4. System retrieves relevant parts
    5. AI generates answers from those parts
    
    #### 💾 Storage
    - Documents persist in vector database
    - Chat history saved locally
    - Everything reloads when you return
    
    #### ⚡ Free Tier Optimized
    - Uses lightweight multilingual models
    - Persistent local storage (no paid databases)
    - Works perfectly on Streamlit Cloud free tier
    """)

st.divider()
st.markdown("<p style='text-align: center; color: #999; font-size: 0.9em;'>Built with 💜 using Streamlit, HuggingFace & Chroma | Persistent • Multilingual • Lightweight</p>", unsafe_allow_html=True)
