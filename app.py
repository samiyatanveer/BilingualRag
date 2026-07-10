import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal styling only — no hardcoded bubble colors, so Streamlit's own
# light/dark theme handles text contrast correctly (this was the bug before).
st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .header-main {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: var(--text-color-secondary, #888); margin-top: 0; }
    </style>
""", unsafe_allow_html=True)

CHROMA_DIR = Path("./chroma_data")
CHROMA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = Path("./chat_history.json")

# ---------- Session state ----------
defaults = {
    "messages": [],
    "documents_loaded": False,
    "db": None,
    "collection": None,
    "history_loaded": False,   # prevents re-reading the file every rerun
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
        st.session_state.db = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            st.session_state.collection = st.session_state.db.get_collection(name="documents")
            st.session_state.documents_loaded = st.session_state.collection.count() > 0
        except Exception:
            st.session_state.collection = st.session_state.db.create_collection(
                name="documents", metadata={"hnsw:space": "cosine"}
            )


def load_chat_history():
    # Only load from disk once per session — reloading on every rerun was
    # harmless for correctness but wasteful, and masked the real duplicate bug
    # (near-identical overlapping chunks, fixed below).
    if not st.session_state.history_loaded:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
        st.session_state.history_loaded = True


def save_chat_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)


def chunk_document(text, chunk_size=800, overlap=100):
    # Old settings (400/50) meant ~87% overlap between consecutive chunks,
    # so retrieved "top 3" results were almost the same text 3 times.
    # 800/100 keeps useful continuity (~12% overlap) without the duplication.
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def add_documents_to_db(documents_text):
    chunks = chunk_document(documents_text)
    embeddings = embeddings_model.encode(chunks, convert_to_numpy=True).tolist()
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
        results = st.session_state.collection.query(query_texts=[question], n_results=top_k)
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


def answer_question(question, context):
    if not context:
        return "No relevant documents found. Please upload documents first."
    full_context = " ".join(context)
    try:
        result = qa_model(question=question, context=full_context)
        return result["answer"]
    except Exception:
        return "Could not generate an answer. Try rephrasing the question."


init_chroma_db()
load_chat_history()

# ---------- Sidebar: documents live here, always visible next to chat ----------
with st.sidebar:
    st.markdown("### 📄 Documents")

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file and st.button("📥 Load file", use_container_width=True, type="primary"):
        text_content = uploaded_file.read().decode("utf-8", errors="ignore")
        with st.spinner("Processing..."):
            num_chunks = add_documents_to_db(text_content)
        st.success(f"Loaded {num_chunks} chunks")

    st.markdown("**Or paste text**")
    document_text = st.text_area(
        "Paste document text",
        height=150,
        label_visibility="collapsed",
        placeholder="Your text here...\n\nآپ رومن اردو میں بھی پیسٹ کر سکتے ہیں"
    )
    if document_text and st.button("📥 Load pasted text", use_container_width=True, type="primary"):
        with st.spinner("Processing..."):
            num_chunks = add_documents_to_db(document_text)
        st.success(f"Loaded {num_chunks} chunks")

    st.divider()

    if st.session_state.documents_loaded:
        st.success(f"✅ {st.session_state.collection.count()} chunks indexed")
    else:
        st.warning("No documents loaded yet")

    if st.session_state.documents_loaded and st.button("🗑️ Clear documents", use_container_width=True):
        st.session_state.db.delete_collection("documents")
        st.session_state.collection = st.session_state.db.create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
        st.session_state.documents_loaded = False
        st.rerun()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        save_chat_history()
        st.rerun()

    with st.expander("📊 System info"):
        st.metric("Chat messages", len(st.session_state.messages))
        st.caption("Multilingual embeddings + extractive QA")

# ---------- Main area: chat, using Streamlit's native chat UI ----------
st.markdown('<p class="header-main">📚 Document Q&A</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask questions about your documents in any language — '
    'English, Urdu, Roman Urdu, Hinglish, etc.</p>',
    unsafe_allow_html=True
)

# Render prior conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("📍 Sources"):
                for j, chunk in enumerate(msg["chunks"], 1):
                    st.caption(f"Source {j}")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)

# Chat input — disabled until documents are loaded
placeholder = (
    "What is... / یہ کیا ہے... / Iska matlab kya hai..."
    if st.session_state.documents_loaded
    else "Upload a document in the sidebar first"
)
prompt = st.chat_input(placeholder, disabled=not st.session_state.documents_loaded)

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "time": datetime.now().isoformat()}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching & generating answer..."):
            chunks = retrieve_relevant_chunks(prompt)
            answer = answer_question(prompt, chunks)
        st.markdown(answer)
        if chunks:
            with st.expander("📍 Sources"):
                for j, chunk in enumerate(chunks, 1):
                    st.caption(f"Source {j}")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks, "time": datetime.now().isoformat()}
    )
    save_chat_history()
