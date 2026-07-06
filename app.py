import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

st.set_page_config(page_title="PDF RAG Lab", layout="wide", page_icon="📄")

st.markdown("""
    <style>
   .big-font {font-size:30px!important; font-weight: bold; color: #4F46E5;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">📄 PDF Chunk + Embedding + Retrieval</p>', unsafe_allow_html=True)
st.write("1 PDF upload karo → Chunks banao → Embedding → Search karo")

# Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.chunks = None
    st.session_state.hybrid_retriever = None
    st.session_state.doc_name = None

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Step 1: Upload PDF")
    uploaded_file = st.file_uploader("PDF choose karo", type="pdf")

    if st.button("🚀 Process PDF", use_container_width=True, type="primary") and uploaded_file:
        with st.spinner("PDF process ho raha hai... 30 sec lagega"):
            # Temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # 1. LOAD
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # 2. CHUNK
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            # 3. EMBEDDING - CPU pe chalega
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            # 4. VECTORSTORE - Memory me
            vectorstore = Chroma.from_documents(chunks, embeddings)

            # 5. HYBRID RETRIEVER
            semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k = 3
            hybrid_retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever], weights=[0.7, 0.3]
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks
            st.session_state.hybrid_retriever = hybrid_retriever
            st.session_state.doc_name = uploaded_file.name
            os.remove(tmp_path)
            st.success(f"✅ Done! {len(docs)} pages → {len(chunks)} chunks")

# MAIN AREA
if st.session_state.chunks:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📊 Stats: {st.session_state.doc_name}")
        st.metric("Total Pages", len(set([d.metadata['page'] for d in st.session_state.chunks])))
        st.metric("Total Chunks", len(st.session_state.chunks))

        with st.expander("👀 Sample Chunk Dekho"):
            st.text(st.session_state.chunks[0].page_content)

    with col2:
        st.subheader("🔍 Step 2: Retrieval Test")
        query = st.text_input("Apna sawal likho:")
        search_type = st.radio("Search Type:", ["Semantic Search", "Hybrid Search"], horizontal=True)

        if st.button("Search", use_container_width=True):
            if query:
                with st.spinner("Searching..."):
                    if search_type == "Semantic Search":
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    else:
                        retriever = st.session_state.hybrid_retriever

                    results = retriever.get_relevant_documents(query)

                    st.success(f"Top {len(results)} relevant chunks mile")
                    for i, doc in enumerate(results):
                        with st.expander(f"Result {i+1} - Page {doc.metadata.get('page', 'N/A')}"):
                            st.write(doc.page_content)
else:
    st.info("👈 Pehle sidebar se PDF upload karke Process karo")

st.markdown("---")
st.caption("Made with LangChain + ChromaDB + Streamlit")
