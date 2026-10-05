import os
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Accounting AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FOLDERS AND PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploaded_files")
CHROMA_FOLDER = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)


# ============================================================
# OLLAMA SETTINGS
# ============================================================

OLLAMA_URL = "http://localhost:11434"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:4b"


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

with st.sidebar:

    st.title("📚 Accounting AI")

    st.markdown("---")

    st.subheader("📄 Add Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # --------------------------------------------
    # SAVE UPLOADED FILES
    # --------------------------------------------

    if uploaded_files:

        current_upload_signature = hashlib.md5(
            "|".join(
                f"{file.name}_{file.size}"
                for file in uploaded_files
            ).encode()
        ).hexdigest()

        previous_signature = st.session_state.get(
            "upload_signature",
            ""
        )

        if current_upload_signature != previous_signature:

            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    UPLOAD_FOLDER,
                    uploaded_file.name
                )

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.session_state.upload_signature = (
                current_upload_signature
            )

            # Clear old cached RAG
            st.cache_resource.clear()

            st.success(
                f"✅ {len(uploaded_files)} document(s) uploaded!"
            )


    st.markdown("---")

    st.subheader("📚 Documents in Knowledge Base")


    # --------------------------------------------
    # FIND ALL PDF DOCUMENTS
    # --------------------------------------------

    pdf_files = []

    # Main accounting.pdf
    main_pdf = os.path.join(
        BASE_DIR,
        "accounting.pdf"
    )

    if os.path.exists(main_pdf):

        pdf_files.append(main_pdf)


    # Uploaded PDFs
    if os.path.exists(UPLOAD_FOLDER):

        for file_name in os.listdir(UPLOAD_FOLDER):

            if file_name.lower().endswith(".pdf"):

                file_path = os.path.join(
                    UPLOAD_FOLDER,
                    file_name
                )

                if file_path not in pdf_files:

                    pdf_files.append(file_path)


    # --------------------------------------------
    # DISPLAY DOCUMENTS
    # --------------------------------------------

    if pdf_files:

        for pdf_file in pdf_files:

            st.write(
                "📄 " + os.path.basename(pdf_file)
            )

    else:

        st.info("No PDF documents uploaded yet.")


    st.markdown("---")

    st.subheader("⚙️ RAG Settings")

    st.write("Chunk size: 2000")
    st.write("Chunk overlap: 0")
    st.write("Retrieved documents: 3")

    st.markdown("---")


    # --------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">📚 Accounting AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your accounting documents.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# GET ALL PDF FILES
# ============================================================

def get_pdf_files():

    files = []

    # Main accounting.pdf
    main_pdf = os.path.join(
        BASE_DIR,
        "accounting.pdf"
    )

    if os.path.exists(main_pdf):

        files.append(main_pdf)


    # Uploaded PDFs
    if os.path.exists(UPLOAD_FOLDER):

        for file_name in os.listdir(UPLOAD_FOLDER):

            if file_name.lower().endswith(".pdf"):

                file_path = os.path.join(
                    UPLOAD_FOLDER,
                    file_name
                )

                if file_path not in files:

                    files.append(file_path)


    return files


# ============================================================
# CREATE DOCUMENT SIGNATURE
# ============================================================

def get_document_signature():

    pdf_files = get_pdf_files()

    signature_data = []

    for file_path in pdf_files:

        modification_time = os.path.getmtime(file_path)

        file_size = os.path.getsize(file_path)

        signature_data.append(
            f"{file_path}_{modification_time}_{file_size}"
        )

    signature = "|".join(signature_data)

    return hashlib.md5(
        signature.encode()
    ).hexdigest()


# ============================================================
# LOAD DOCUMENTS
# ============================================================

@st.cache_resource
def load_documents(document_signature):

    pdf_files = get_pdf_files()

    if not pdf_files:

        raise FileNotFoundError(
            "No PDF documents found. "
            "Please upload at least one PDF."
        )


    all_documents = []


    for pdf_path in pdf_files:

        try:

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            all_documents.extend(documents)

        except Exception as e:

            st.warning(
                f"Could not load {os.path.basename(pdf_path)}: {e}"
            )


    if not all_documents:

        raise ValueError(
            "No readable PDF documents were found."
        )


    return all_documents


# ============================================================
# SPLIT DOCUMENTS
# ============================================================

@st.cache_resource
def create_chunks(document_signature):

    documents = load_documents(
        document_signature
    )


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )


    chunks = splitter.split_documents(
        documents
    )


    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

@st.cache_resource
def create_embeddings():

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    return embeddings


# ============================================================
# CREATE VECTOR STORE
# ============================================================

@st.cache_resource
def create_vectorstore(document_signature):

    chunks = create_chunks(
        document_signature
    )

    embeddings = create_embeddings()


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER,
        collection_name="accounting_documents"
    )


    return vectorstore


# ============================================================
# CREATE RETRIEVER
# ============================================================

@st.cache_resource
def create_retriever(document_signature):

    vectorstore = create_vectorstore(
        document_signature
    )


    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3
        }
    )


    return retriever


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def create_llm():

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.2
    )

    return llm


# ============================================================
# INITIALIZE CHAT
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "💬 Ask an accounting question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if query:

    # --------------------------------------------
    # SHOW USER QUESTION
    # --------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------
    # AI RESPONSE
    # --------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your documents..."
        ):

            try:

                # --------------------------------
                # DOCUMENT SIGNATURE
                # --------------------------------

                document_signature = (
                    get_document_signature()
                )


                # --------------------------------
                # CHECK DOCUMENTS
                # --------------------------------

                if not get_pdf_files():

                    st.error(
                        "❌ No PDF documents found. "
                        "Please upload a PDF from the sidebar."
                    )

                    st.stop()


                # --------------------------------
                # CREATE RETRIEVER
                # --------------------------------

                retriever = create_retriever(
                    document_signature
                )


                # --------------------------------
                # SEARCH DOCUMENTS
                # --------------------------------

                results = retriever.invoke(
                    query
                )


                # --------------------------------
                # CREATE CONTEXT
                # --------------------------------

                context_parts = []


                for i, result in enumerate(results):

                    context_parts.append(
                        f"""
--- SOURCE {i + 1} ---

{result.page_content}
"""
                    )


                context = "\n".join(
                    context_parts
                )


                # --------------------------------
                # CREATE PROMPT
                # --------------------------------

                prompt = f"""
You are an Accounting AI Assistant.

Answer the user's question using ONLY the
information provided in the document context.

Give a clear, simple and easy-to-understand
answer.

Do not invent information.

If the answer is not available in the
documents, say:

"I couldn't find this information in the
provided accounting documents."

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""


                # --------------------------------
                # LOAD LLM
                # --------------------------------

                llm = create_llm()


                # --------------------------------
                # GENERATE ANSWER
                # --------------------------------

                response = llm.invoke(
                    prompt
                )


                answer = response.content


                # --------------------------------
                # DISPLAY ANSWER
                # --------------------------------

                st.markdown(answer)


                # --------------------------------
                # SHOW SOURCES
                # --------------------------------

                with st.expander(
                    "📄 View retrieved sources"
                ):

                    for i, result in enumerate(
                        results
                    ):

                        page_number = (
                            result.metadata.get(
                                "page",
                                "Unknown"
                            )
                        )


                        if isinstance(
                            page_number,
                            int
                        ):

                            page_display = (
                                page_number + 1
                            )

                        else:

                            page_display = (
                                page_number
                            )


                        source_file = (
                            result.metadata.get(
                                "source",
                                "Unknown"
                            )
                        )


                        source_name = (
                            os.path.basename(
                                source_file
                            )
                            if source_file != "Unknown"
                            else "Unknown"
                        )


                        st.markdown(
                            f"""
**Source {i + 1}**

📄 **Document:** {source_name}

📑 **Page:** {page_display}
"""
                        )


                        st.write(
                            result.page_content[:1000]
                        )


                        st.markdown("---")


                # --------------------------------
                # SAVE ANSWER
                # --------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                st.error(
                    "❌ Something went wrong while running the RAG system."
                )

                st.code(
                    str(e)
                )