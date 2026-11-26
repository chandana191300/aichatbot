
"""
RAG Chat with PDFs & Web Search
Updated Version - LangChain 0.2+
"""

import streamlit as st
import os
import tempfile
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

# Load environment variables
load_dotenv()

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="AI Search Chatbot",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Chat with PDFs & Tools")

st.markdown(
    "Ask questions using your PDFs, web search, "
    "academic papers, and Wikipedia."
)

# ==================== API KEY CONFIGURATION ====================

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY", "")

if not api_key:
    st.error(
        "Groq API key is not configured. "
        "Please contact the app administrator."
    )
    st.stop()

# ==================== SIDEBAR ====================

st.sidebar.title("⚙️ Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.info(
    "You can ask questions about uploaded PDFs "
    "or search the web, academic papers, and Wikipedia."
)

# ==================== VECTOR STORE ====================

def load_and_vectorize_pdfs(pdf_files):
    """Load PDF documents and convert them into vectors."""

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
            st.warning("No pages found in the uploaded PDFs.")
            return None

        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            st.warning("No readable text found in the PDFs.")
            return None

        # Generate embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        return vector_store

    except Exception as e:
        st.error(f"Error loading PDFs: {str(e)}")
        return None


# ==================== PDF SEARCH ====================

def search_pdfs(query, vector_store):
    """Search uploaded PDFs using similarity search."""

    if vector_store is None:
        return ""

    try:
        docs = vector_store.similarity_search(
            query,
            k=3
        )

        if not docs:
            return ""

        results = []

        for doc in docs:
            results.append(
                f"📄 {doc.page_content[:500]}"
            )

        return "\n---\n".join(results)

    except Exception as e:
        st.error(f"PDF search error: {str(e)}")
        return ""


# ==================== WEB SEARCH ====================

def search_web(query):
    """Search the internet using DuckDuckGo."""

    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)[:1000]

    except Exception as e:
        return f"Web search unavailable: {str(e)}"


# ==================== ARXIV SEARCH ====================

def search_arxiv(query):
    """Search academic research papers using arXiv."""

    try:
        arxiv = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(
                top_k_results=2
            )
        )

        return arxiv.run(query)[:1000]

    except Exception as e:
        return f"arXiv search unavailable: {str(e)}"


# ==================== WIKIPEDIA SEARCH ====================

def search_wikipedia(query):
    """Search Wikipedia."""

    try:
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(
                top_k_results=2
            )
        )

        return wiki.run(query)[:1000]

    except Exception as e:
        return f"Wikipedia search unavailable: {str(e)}"


# ==================== CHAT SESSION ====================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None


# ==================== PDF PROCESSING ====================

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


# ==================== DISPLAY CHAT HISTORY ====================

st.markdown("### 💬 Chat")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ==================== USER INPUT ====================

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

            st.write("🔍 Searching information...")

            # Search all sources
            with st.spinner("Searching PDFs and external sources..."):

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
                    f"From your PDFs:\n"
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

            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=api_key,
                model="openai/gpt-oss-20b",
                temperature=0.7
            )

            # System instructions
            system_msg = SystemMessage(
                content="""
                You are a helpful AI search assistant.

                Answer questions clearly and accurately
                using the provided context.

                Use relevant information from PDFs,
                web search, academic papers, and Wikipedia.

                If information comes from search results,
                mention the relevant source.

                If the context does not contain the answer,
                clearly explain that.

                Be concise, informative, and helpful.
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
