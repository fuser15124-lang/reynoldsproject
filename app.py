
import os
import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -------------------- Configuration --------------------

st.set_page_config(
    page_title="Smart PDF Assistant | دستیار هوشمند PDF",
    page_icon="🌲",
    layout="wide",
)

# Read the API key from Streamlit Secrets or environment variables.
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    GROQ_API_KEY = ""

GROQ_API_KEY = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

# -------------------- Sidebar --------------------

with st.sidebar:
    st.markdown("### 🌐 Language / انتخاب زبان")
    language = st.selectbox(
        "Language",
        ["فارسی", "English"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🌗 Theme Mode / حالت نمایش")
    theme_mode = st.radio(
        "Theme",
        ["Dark Mode (کیهانی خفن)", "Light Mode (جنگل طبیعی و واضح)"],
        label_visibility="collapsed",
    )

if "Dark" in theme_mode:
    bg_style = """
    .stApp {
        background-image:
            linear-gradient(rgba(5, 8, 20, 0.75), rgba(10, 15, 30, 0.82)),
            url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=2560&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-header, .stChatMessage {
        background: rgba(15, 23, 42, 0.88) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 16px;
        color: #f8fafc !important;
    }
    .main-header h1, .main-header p, .stChatMessage p {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
    }
    """
else:
    bg_style = """
    .stApp {
        background-image:
            linear-gradient(rgba(255,255,255,0.25), rgba(255,255,255,0.35)),
            url('https://images.unsplash.com/photo-1448375240586-882707db888b?q=80&w=2560&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-header, .stChatMessage {
        background: rgba(255,255,255,0.94) !important;
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0,0,0,0.15);
        border-radius: 16px;
        color: #0f172a !important;
    }
    .main-header h1, .main-header p, .stChatMessage p {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f4fbf4;
    }
    """

st.markdown(
    f"""
    <style>
    {bg_style}
    .main-header {{
        padding: 2.4rem;
        border-radius: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
        margin-bottom: 2rem;
        border-right: 6px solid #059669;
    }}
    .stChatMessage {{
        padding: 1.2rem;
        margin-bottom: 12px;
    }}
    .stButton > button {{
        border-radius: 12px;
        font-weight: 600;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Translations --------------------

texts = {
    "English": {
        "title": "✨ Smart PDF Chat Assistant 🌲",
        "subtitle": (
            "Upload your PDF and ask questions using AI-powered "
            "document search."
        ),
        "sidebar_header": "📄 Document Management",
        "upload_label": "Choose a PDF file",
        "process_btn": "🚀 Process & Analyze",
        "processing_spinner": "Extracting text and preparing AI...",
        "success_msg": "✅ File processed successfully! Ask your question.",
        "chat_input": "Type your question here...",
        "warning_no_file": "⚠️ Please upload and process a PDF first.",
        "searching_spinner": "Searching and generating an answer...",
        "empty_pdf": "No extractable text was found in this PDF.",
        "missing_key": "GROQ_API_KEY is not configured in Secrets.",
        "error": "An error occurred. Check the app logs for details.",
    },
    "فارسی": {
        "title": "✨ دستیار هوشمند گفتگو با فایل‌های PDF 🌲",
        "subtitle": (
            "فایل PDF خود را آپلود کنید و با هوش مصنوعی "
            "از محتوای آن سؤال بپرسید."
        ),
        "sidebar_header": "📄 مدیریت سند",
        "upload_label": "یک فایل PDF انتخاب کنید",
        "process_btn": "🚀 پردازش و تحلیل فایل",
        "processing_spinner": "در حال استخراج متن و آماده‌سازی...",
        "success_msg": "✅ فایل با موفقیت پردازش شد! سؤال خود را بپرسید.",
        "chat_input": "سؤال خود را اینجا بنویسید...",
        "warning_no_file": "⚠️ ابتدا یک فایل PDF آپلود و پردازش کنید.",
        "searching_spinner": "در حال جست‌وجو و تولید پاسخ...",
        "empty_pdf": "متن قابل استخراجی در این فایل PDF پیدا نشد.",
        "missing_key": "کلید GROQ_API_KEY در Secrets تنظیم نشده است.",
        "error": "خطایی رخ داد. برای جزئیات بیشتر Logs را بررسی کنید.",
    },
}

t = texts[language]

st.markdown(
    f"""
    <div class="main-header">
        <h1 style="margin-bottom:8px;font-size:26px;">
            {t["title"]}
        </h1>
        <p style="font-size:15px;margin:0;line-height:1.5;">
            {t["subtitle"]}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Session State --------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# -------------------- PDF Processing --------------------

with st.sidebar:
    st.markdown("---")
    st.markdown(f"### {t['sidebar_header']}")

    uploaded_file = st.file_uploader(
        t["upload_label"],
        type=["pdf"],
    )

    if uploaded_file is not None:
        file_signature = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if st.button(t["process_btn"], use_container_width=True):
            try:
                with st.spinner(t["processing_spinner"]):
                    pdf_reader = PdfReader(uploaded_file)

                    extracted_pages = []
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_pages.append(page_text)

                    document_text = "\n\n".join(extracted_pages)

                    if not document_text.strip():
                        st.session_state.vector_store = None
                        st.error(t["empty_pdf"])
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                        )
                        chunks = splitter.split_text(document_text)

                        if st.session_state.embeddings is None:
                            st.session_state.embeddings = (
                                HuggingFaceEmbeddings(
                                    model_name="all-MiniLM-L6-v2"
                                )
                            )

                        vector_store = FAISS.from_texts(
                            chunks,
                            embedding=st.session_state.embeddings,
                        )

                        st.session_state.vector_store = vector_store
                        st.session_state.processed_file = file_signature
                        st.session_state.chat_history = []

                        st.success(t["success_msg"])

            except Exception:
                st.error(t["error"])
                st.exception(Exception("PDF processing failed. Check logs."))

# -------------------- Chat History --------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- RAG Chat --------------------

user_query = st.chat_input(t["chat_input"])

if user_query:
    if st.session_state.vector_store is None:
        st.warning(t["warning_no_file"])

    elif not GROQ_API_KEY:
        st.error(t["missing_key"])

    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner(t["searching_spinner"]):
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 3}
                    )

                    # Current replacement model; verify availability
                    # in the Groq Console if necessary.
                    llm = ChatGroq(
                        model="openai/gpt-oss-120b",
                        groq_api_key=GROQ_API_KEY,
                        temperature=0.3,
                    )

                    prompt = ChatPromptTemplate.from_template(
                        """
You are a helpful AI assistant.
Answer using only the supplied document context.

- If the user asks in Persian, answer in fluent Persian.
- If the user asks in English, answer in fluent English.
- If the answer is not supported by the context, say:
  "در فایل یافت نشد / Not found in the file".

Context:
{context}

Question:
{input}

Answer:
"""
                    )

                    def format_docs(docs):
                        return "\n\n".join(
                            doc.page_content for doc in docs
                        )

                    rag_chain = (
                        {
                            "context": retriever | format_docs,
                            "input": RunnablePassthrough(),
                        }
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    answer = rag_chain.invoke(user_query)

                st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception:
                st.error(t["error"])
                st.exception(
                    Exception(
                        "Request failed. Check Streamlit Cloud Logs "
                        "for the original exception."
                    )
                )
