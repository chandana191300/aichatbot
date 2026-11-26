
"""
AI Search Chatbot - RAG with PDFs & Web Search
LangChain 0.2+ compatible
"""

import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    ArxivQueryRun,
    WikipediaQueryRun
)

from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)

from langchain_core.messages import HumanMessage, SystemMessage


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Search Chatbot",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Chat with PDFs & Tools")

st.markdown(
    """
    Ask questions using your PDFs, web search,
    academic research papers, and Wikipedia.
    """
)


# =====================================================
# 2. API KEY CONFIGURATION
# =====================================================

load_dotenv()

def get_groq_api_key():
    """Read Groq API key from Streamlit Secrets or local .env."""

    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")

    return api_key


api_key = get_groq_api_key()

if not api_key:
    st.error(
        "Groq API key is not configured. "
        "Please contact the app administrator."
    )
    st.stop()


# =====================================================
# 3. INITIALIZE LLM
# =====================================================

@st.cache_resource
def initialize_llm(key):
    return ChatGroq(
        groq_api_key=key,
        model="openai/gpt-oss-20b",
        temperature=0.7
    )


llm = initialize_llm(api_key)


# =====================================================
# 4. INITIALIZE EMBEDDINGS
# =====================================================

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# =====================================================
# 5. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    You can ask questions about:

    - Uploaded PDF documents
    - Internet search results
    - Academic research papers
    - Wikipedia articles
    """
)


# =====================================================
# 6. LOAD AND VECTORIZE PDF DOCUMENTS
# =====================================================

def load_and_vectorize_pdfs(pdf_files):
    """Load PDFs, split text into chunks and create FAISS vectors."""

    try:
        documents = []

        for pdf_file in pdf_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if not documents:
            st.warning("No pages found in uploaded PDFs.")
            return None

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            st.warning("No readable text found in PDFs.")
            return None

        # Generate embeddings
        embeddings = initialize_embeddings()

        # Create FAISS vector database
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vector_store

    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return None


# =====================================================
# 7. PDF SEARCH
# =====================================================

def search_pdfs(query, vector_store):
    """Search PDF documents using similarity search."""

    if vector_store is None:
        return ""

    try:
        docs = vector_store.similarity_search(
            query,
            k=3
        )

        results = []

        for doc in docs:
            source = doc.metadata.get("source", "Uploaded PDF")
            page = doc.metadata.get("page", "Unknown")

            results.append(
                f"Source: {source}, Page: {page + 1 if isinstance(page, int) else page}\n"
                f"{doc.page_content[:700]}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"PDF search error: {str(e)}"


# =====================================================
# 8. WEB SEARCH
# =====================================================

@st.cache_resource
def initialize_web_search():
    return DuckDuckGoSearchRun()


def search_web(query):
    """Search the internet using DuckDuckGo."""

    try:
        search = initialize_web_search()
        return search.run(query)[:1500]

    except Exception as e:
        return f"Web search unavailable: {str(e)}"


# =====================================================
# 9. ARXIV SEARCH
# =====================================================

@st.cache_resource
def initialize_arxiv():
    return ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(
            top_k_results=2
        )
    )


def search_arxiv(query):
    """Search academic research papers."""

    try:
        arxiv = initialize_arxiv()
        return arxiv.run(query)[:1500]

    except Exception as e:
        return f"arXiv search unavailable: {str(e)}"


# =====================================================
# 10. WIKIPEDIA SEARCH
# =====================================================

@st.cache_resource
def initialize_wikipedia():
    return WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2
        )
    )


def search_wikipedia(query):
    """Search Wikipedia."""

    try:
        wiki = initialize_wikipedia()
        return wiki.run(query)[:1500]

    except Exception as e:
        return f"Wikipedia search unavailable: {str(e)}"


# =====================================================
# 11. CHAT SESSION
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None


# =====================================================
# 12. PDF PROCESSING
# =====================================================

if uploaded_files:

    current_signature = tuple(
        (
            pdf.name,
            len(pdf.getvalue()),
            hash(pdf.getvalue())
        )
        for pdf in uploaded_files
    )

    if current_signature != st.session_state.pdf_signature:

        with st.spinner("📚 Processing PDF documents..."):

            st.session_state.vector_store = (
                load_and_vectorize_pdfs(uploaded_files)
            )

            st.session_state.pdf_signature = current_signature

        if st.session_state.vector_store:
            st.success(
                f"Successfully processed "
                f"{len(uploaded_files)} PDF(s)."
            )

else:
    st.session_state.vector_store = None
    st.session_state.pdf_signature = None


# =====================================================
# 13. DISPLAY CHAT HISTORY
# =====================================================

st.markdown("### 💬 Chat")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =====================================================
# 14. USER INPUT
# =====================================================

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    st.chat_message("user").write(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):

        try:

            # Search information sources
            with st.spinner(
                "🔍 Searching PDFs and external sources..."
            ):

                pdf_results = search_pdfs(
                    user_input,
                    st.session_state.vector_store
                )

                web_results = search_web(user_input)

                arxiv_results = search_arxiv(user_input)

                wiki_results = search_wikipedia(user_input)

            # Build context
            context = ""

            if pdf_results:
                context += (
                    f"From uploaded PDFs:\n"
                    f"{pdf_results}\n\n"
                )

            if web_results:
                context += (
                    f"From Web Search:\n"
                    f"{web_results}\n\n"
                )

            if arxiv_results:
                context += (
                    f"From Academic Papers:\n"
                    f"{arxiv_results}\n\n"
                )

            if wiki_results:
                context += (
                    f"From Wikipedia:\n"
                    f"{wiki_results}\n\n"
                )

            # System instructions
            system_msg = SystemMessage(
                content="""
                You are a helpful AI search assistant.

                Answer questions clearly and accurately
                using the provided context.

                Use relevant information from PDFs,
                web search, academic papers and Wikipedia.

                Mention relevant sources when available.

                If the context does not contain the answer,
                clearly explain that.

                Be concise, informative and helpful.
                """
            )

            # User message with context
            if context:

                user_msg = HumanMessage(
                    content=(
                        f"Context:\n{context}\n\n"
                        f"Question: {user_input}"
                    )
                )

            else:

                user_msg = HumanMessage(
                    content=user_input
                )

            # Generate answer
            with st.spinner("💭 Generating response..."):

                response = llm.invoke(
                    [system_msg, user_msg]
                )

            answer = response.content

            # Display answer
            st.write(answer)

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            error_msg = f"❌ Error: {str(e)}"

            st.error(error_msg)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg
                }
            )
