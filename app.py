import os
import re
import logging
from typing import List

import streamlit as st
from pypdf import PdfReader

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Configuration --------------------

st.set_page_config(
    page_title="Smart PDF & Web Assistant | دستیار هوشمند PDF و وب",
    page_icon="🌲",
    layout="wide",
)

# Read the secret from Streamlit Cloud Secrets first, then environment variables.
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    GROQ_API_KEY = ""
GROQ_API_KEY = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

# A production model listed in Groq's current model catalog.
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

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
        background-image: linear-gradient(rgba(5,8,20,.78), rgba(10,15,30,.84)),
        url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=2560&auto=format&fit=crop');
        background-size: cover; background-position: center; background-attachment: fixed;
    }
    .main-header, .stChatMessage {
        background: rgba(15,23,42,.88) !important; backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,.15); border-radius: 16px;
        color: #f8fafc !important;
    }
    .main-header h1, .main-header p, .stChatMessage p { color: #f8fafc !important; }
    [data-testid="stSidebar"] { background-color: rgba(15,23,42,.95); }
    """
else:
    bg_style = """
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,.28), rgba(255,255,255,.38)),
        url('https://images.unsplash.com/photo-1448375240586-882707db888b?q=80&w=2560&auto=format&fit=crop');
        background-size: cover; background-position: center; background-attachment: fixed;
    }
    .main-header, .stChatMessage {
        background: rgba(255,255,255,.94) !important; backdrop-filter: blur(16px);
        border: 1px solid rgba(0,0,0,.15); border-radius: 16px;
        color: #0f172a !important;
    }
    .main-header h1, .main-header p, .stChatMessage p { color: #0f172a !important; }
    [data-testid="stSidebar"] { background-color: #f4fbf4; }
    """

st.markdown(
    f"""
    <style>
    {bg_style}
    .main-header {{
        padding: 2.2rem; border-radius: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,.22);
        margin-bottom: 2rem; border-right: 6px solid #059669;
    }}
    .stChatMessage {{ padding: 1.1rem; margin-bottom: 12px; }}
    .stButton > button {{
        border-radius: 12px; font-weight: 600;
        background: linear-gradient(135deg,#059669,#10b981);
        color: white; border: none; padding: .6rem 1.2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Translations --------------------

texts = {
    "English": {
        "title": "✨ Smart PDF & Web Chat Assistant 🌲",
        "subtitle": "Upload a PDF, ask questions, solve exercises, or chat naturally.",
        "sidebar_header": "📄 Document Management",
        "upload_label": "Choose a PDF file",
        "process_btn": "🚀 Process & Analyze",
        "processing_spinner": "Extracting text and indexing document...",
        "success_msg": "✅ PDF processed successfully. You can ask questions now.",
        "chat_input": "Type your message, question, or exercise here...",
        "warning_no_file": "No PDF has been processed yet. You can still ask general questions.",
        "empty_pdf": "No extractable text was found in this PDF. Scanned PDFs may need OCR.",
        "missing_key": "GROQ_API_KEY is not configured in Streamlit Secrets or environment variables.",
        "thinking": "Thinking and searching...",
        "pdf_ready": "Processed PDF",
        "new_file_warning": "You selected a different PDF. Click Process & Analyze before asking about it.",
        "web_note": "Web search was unavailable for this request; answering from available context.",
        "error": "The request failed. Details are shown below; remove any secrets before sharing logs.",
        "process_first": "Please process the selected PDF first.",
    },
    "فارسی": {
        "title": "✨ دستیار هوشمند PDF و وب 🌲",
        "subtitle": "فایل PDF آپلود کنید، سؤال بپرسید، تمرین حل کنید یا گفتگو کنید.",
        "sidebar_header": "📄 مدیریت سند",
        "upload_label": "یک فایل PDF انتخاب کنید",
        "process_btn": "🚀 پردازش و تحلیل فایل",
        "processing_spinner": "در حال استخراج متن و ایندکس‌سازی فایل...",
        "success_msg": "✅ فایل با موفقیت پردازش شد؛ حالا سؤال خود را بپرسید.",
        "chat_input": "پیام، سؤال یا تمرین خود را اینجا بنویسید...",
        "warning_no_file": "هنوز فایلی پردازش نشده است؛ می‌توانید سؤال عمومی هم بپرسید.",
        "empty_pdf": "متن قابل استخراجی در PDF پیدا نشد. برای PDF اسکن‌شده ممکن است OCR لازم باشد.",
        "missing_key": "کلید GROQ_API_KEY در Secrets یا متغیرهای محیطی تنظیم نشده است.",
        "thinking": "در حال بررسی سؤال و جست‌وجو...",
        "pdf_ready": "فایل پردازش‌شده",
        "new_file_warning": "فایل جدیدی انتخاب کرده‌اید؛ قبل از سؤال درباره آن، روی پردازش و تحلیل بزنید.",
        "web_note": "جست‌وجوی وب در دسترس نبود؛ پاسخ بر اساس اطلاعات موجود تولید می‌شود.",
        "error": "در اجرای درخواست خطا رخ داد. جزئیات پایین نمایش داده شده‌اند؛ قبل از اشتراک‌گذاری، اطلاعات محرمانه را حذف کنید.",
        "process_first": "لطفاً ابتدا فایل انتخاب‌شده را پردازش کنید.",
    },
}
t = texts[language]

st.markdown(
    f"""
    <div class="main-header">
        <h1 style="margin-bottom:8px;font-size:26px;">{t["title"]}</h1>
        <p style="font-size:15px;margin:0;line-height:1.5;">{t["subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Session State --------------------

defaults = {
    "chat_history": [],
    "vector_store": None,
    "embeddings": None,
    "processed_file": None,
    "page_texts": {},
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------------- PDF Processing --------------------

with st.sidebar:
    st.markdown("---")
    st.markdown(f"### {t['sidebar_header']}")
    uploaded_file = st.file_uploader(t["upload_label"], type=["pdf"])

    current_signature = (
        (uploaded_file.name, uploaded_file.size) if uploaded_file is not None else None
    )

    if uploaded_file is not None and st.button(t["process_btn"], use_container_width=True):
        try:
            with st.spinner(t["processing_spinner"]):
                reader = PdfReader(uploaded_file)
                page_documents: List[Document] = []
                complete_page_texts = {}

                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        complete_page_texts[page_number] = page_text.strip()
                        page_documents.append(
                            Document(
                                page_content=page_text.strip(),
                                metadata={
                                    "page": page_number,
                                    "source": uploaded_file.name,
                                },
                            )
                        )

                if not page_documents:
                    st.session_state.vector_store = None
                    st.session_state.processed_file = None
                    st.session_state.page_texts = {}
                    st.error(t["empty_pdf"])
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1200,
                        chunk_overlap=250,
                        add_start_index=True,
                    )
                    chunks = splitter.split_documents(page_documents)

                    if st.session_state.embeddings is None:
                        st.session_state.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )

                    st.session_state.vector_store = FAISS.from_documents(
                        chunks,
                        embedding=st.session_state.embeddings,
                    )
                    st.session_state.processed_file = current_signature
                    st.session_state.page_texts = complete_page_texts
                    st.session_state.chat_history = []
                    st.success(t["success_msg"])

        except Exception as exc:
            logger.exception("PDF processing failed")
            st.error(t["error"])
            st.exception(exc)

# -------------------- Chat History --------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- Helper Functions --------------------

def is_simple_greeting(query: str) -> bool:
    """Match only short, standalone greetings; avoid matching words inside questions."""
    normalized = re.sub(r"[^\w\s]", "", query.casefold(), flags=re.UNICODE)
    normalized = " ".join(normalized.split())
    greetings = {
        "سلام", "درود", "سلام علیکم", "صبح بخیر", "عصر بخیر",
        "شب بخیر", "خوبی", "خوب هستی", "حالت چطوره",
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "how are you", "thanks", "thank you",
        "ممنون", "تشکر", "مرسی",
    }
    return normalized in greetings


def build_history(history):
    """Convert Streamlit history into LangChain messages, excluding current user turn."""
    result = []
    for item in history[:-1]:
        if item["role"] == "user":
            result.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            result.append(AIMessage(content=item["content"]))
    return result


def requested_page_number(query: str):
    """Detect explicit page requests in Persian or English."""
    normalized = query.casefold()
    # Normalize Persian and Arabic-Indic digits to ASCII digits.
    normalized = normalized.translate(
        str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
    )
    patterns = (
        r"(?:صفحه|صفحهٔ|صفحه‌ی)\s*(\d+)",
        r"(?:page|p\.)\s*(\d+)",
        r"(\d+)\s*(?:صفحه|page)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def format_retrieved_docs(docs):
    parts = []
    for doc in docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "PDF")
        page_label = f"[Page {page}]" if page else "[PDF page unknown]"
        parts.append(f"{page_label} [Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


# -------------------- Chat Handler --------------------

user_query = st.chat_input(t["chat_input"])

if user_query:
    if not GROQ_API_KEY:
        st.error(t["missing_key"])
    elif (
        uploaded_file is not None
        and st.session_state.processed_file is not None
        and current_signature != st.session_state.processed_file
    ):
        st.warning(t["new_file_warning"])
    else:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            if language == "فارسی":
                st.markdown(
                    f'<div dir="rtl" style="text-align:right; direction:rtl; '
                    f'font-size:17px; line-height:1.95; overflow-wrap:anywhere;">'
                    f'{user_query}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner(t["thinking"]):
                    llm = ChatGroq(
                        model=MODEL_NAME,
                        groq_api_key=GROQ_API_KEY,
                        temperature=0.3,
                    )

                    history = build_history(st.session_state.chat_history)

                    if is_simple_greeting(user_query):
                        prompt = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                "You are a friendly, helpful AI assistant and academic tutor. "
                                "Reply naturally in the same language as the user (Persian or English). "
                                "Keep greetings brief and offer help with PDFs, questions, or exercises.",
                            ),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                        ])
                        chain = (
                            {
                                "chat_history": lambda _: history,
                                "input": RunnablePassthrough(),
                            }
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        answer = chain.invoke(user_query)

                    else:
                        pdf_context = "No PDF has been uploaded and processed."
                        vector_store = st.session_state.vector_store

                        # If the user asks about a specific page, pass the complete extracted
                        # text of that page instead of relying on top-k semantic retrieval.
                        page_number = requested_page_number(user_query)
                        page_texts = st.session_state.page_texts

                        if page_number is not None and page_texts:
                            if page_number in page_texts:
                                pdf_context = (
                                    f"[Complete text of PDF page {page_number}]\\n"
                                    f"{page_texts[page_number]}"
                                )
                            else:
                                available_pages = ", ".join(
                                    str(p) for p in sorted(page_texts.keys())
                                )
                                pdf_context = (
                                    f"The requested PDF page {page_number} was not found. "
                                    f"Pages with extractable text include: {available_pages}."
                                )
                        elif vector_store is not None:
                            retriever = vector_store.as_retriever(
                                search_type="mmr",
                                search_kwargs={"k": 6, "fetch_k": 20},
                            )
                            docs = retriever.invoke(user_query)
                            if docs:
                                pdf_context = format_retrieved_docs(docs)
                            else:
                                pdf_context = "No relevant passages were found in the PDF."

                        # Web search is best-effort; the app still works if it is unavailable.
                        try:
                            web_search = DuckDuckGoSearchRun()
                            web_results = web_search.invoke(user_query)
                        except Exception:
                            logger.exception("DuckDuckGo web search failed")
                            web_results = "Web search unavailable for this request."

                        prompt = ChatPromptTemplate.from_messages([
                            (
                                "system",
                                """You are a careful academic tutor and research assistant.
Answer in the same language as the user (Persian or English).

Use the PDF context first when it contains relevant information. When the user asks to explain,
summarize, or describe a specific page, cover the whole supplied page text, not just one sentence.
For page summaries, include the main idea, key points, important terms/formulas, and conclusion
in a concise, organized answer. Do not claim the page is missing if its full text is supplied.
When you use PDF content, cite page labels shown in the context, such as [Page 3]. Never invent page numbers.
If the PDF does not answer the question, you may use the web search results.
Be clear when a claim comes from web results, and do not claim that search results are verified
unless the supplied results actually support it.
If neither source supports an answer, say so and explain what is missing.
For exercises, show a clear step-by-step solution and check calculations where possible.

PDF context:
{pdf_context}

Web search results:
{web_results}
""",
                            ),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                        ])

                        chain = (
                            {
                                "pdf_context": lambda _: pdf_context,
                                "web_results": lambda _: web_results,
                                "chat_history": lambda _: history,
                                "input": RunnablePassthrough(),
                            }
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        answer = chain.invoke(user_query)

                if language == "فارسی":
                    # Escape HTML so model output is displayed as text, not markup.
                    import html
                    safe_answer = html.escape(answer).replace("\\n", "<br>")
                    st.markdown(
                        f'<div dir="rtl" style="text-align:right; direction:rtl; '
                        f'font-size:17px; line-height:1.95; overflow-wrap:anywhere;">'
                        f'{safe_answer}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                logger.exception("Chat request failed")
                st.error(t["error"])
                st.exception(exc)
