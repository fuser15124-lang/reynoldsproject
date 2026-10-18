import os
import re
import io
import json
import html
import base64
import logging
from typing import List, Dict, Optional

import streamlit as st
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_pdf_assistant")

st.set_page_config(
    page_title="NOVA | دستیار PDF و وب",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

GROQ_API_KEY = get_secret("GROQ_API_KEY")
MODEL_NAME = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GOOGLE_CSE_ID = get_secret("GOOGLE_CSE_ID")
SERPAPI_API_KEY = get_secret("SERPAPI_API_KEY")
VISION_MODEL = get_secret("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Neon dark interface + Persian-friendly typography.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');
:root { --bg:#070b16; --panel:#10172a; --cyan:#35e7ff; --pink:#ff4fd8; --purple:#8b5cf6; }
html, body, [class*="css"], .stApp { font-family:'Vazirmatn','Inter',sans-serif; }
.stApp { background: radial-gradient(ellipse at 10% 0%,rgba(139,92,246,.18),transparent 35%),
 radial-gradient(ellipse at 90% 15%,rgba(53,231,255,.12),transparent 30%), #070b16; color:#eaf2ff; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1425,#090e1b); border-right:1px solid #25324c; }
.main-header { padding:1.5rem 1.7rem; margin:.2rem 0 1.4rem; border:1px solid rgba(53,231,255,.42);
 border-radius:22px; background:linear-gradient(115deg,rgba(16,23,42,.96),rgba(20,17,46,.91));
 box-shadow:0 0 28px rgba(53,231,255,.09), inset 0 0 28px rgba(139,92,246,.05); }
.main-header h1 { font-size:clamp(1.5rem,3vw,2.25rem); margin:0; color:#f7fbff; }
.main-header p { color:#aabbd5; margin:.5rem 0 0; line-height:1.9; }
.stButton>button, .stDownloadButton>button { border:1px solid rgba(53,231,255,.5); border-radius:12px;
 background:linear-gradient(105deg,rgba(53,231,255,.16),rgba(139,92,246,.2)); color:#effcff;
 transition:all .2s; font-weight:600; }
.stButton>button:hover,.stDownloadButton>button:hover { border-color:#ff4fd8; box-shadow:0 0 18px #35e7ff33; color:white; }
[data-testid="stChatMessage"] { background:rgba(16,23,42,.76); border:1px solid #25324c; border-radius:16px; }
[data-testid="stChatInput"] { border-color:#344563; }
[data-testid="stFileUploader"] { border:1px dashed #405775; border-radius:14px; padding:.5rem; }
h1,h2,h3 { letter-spacing:-.02em; }
.neon { color:#35e7ff; text-shadow:0 0 16px #35e7ff44; }
.small-muted { color:#94a8c7; font-size:.88rem; }
div[data-testid="stMetric"] { background:#10172a; padding:12px; border:1px solid #25324c; border-radius:14px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## <span class='neon'>✦ NOVA</span>", unsafe_allow_html=True)
    language = st.selectbox("زبان / Language", ["فارسی", "English"], label_visibility="collapsed")
    mode = st.radio("حالت دستیار", ["گفت‌وگو و تحلیل", "معلم خصوصی", "ساخت آزمون"] if language == "فارسی"
                    else ["Chat & Analyze", "Private Tutor", "Create Quiz"])
    st.markdown("---")
    st.markdown("### 📚 مدیریت اسناد" if language == "فارسی" else "### 📚 Documents")
    uploaded_pdfs = st.file_uploader(
        "چند PDF انتخاب کنید" if language == "فارسی" else "Upload multiple PDFs",
        type=["pdf"], accept_multiple_files=True, key="pdf_upload"
    )
    process_clicked = st.button("⚡ پردازش و ایندکس همه PDFها" if language == "فارسی" else "⚡ Process all PDFs",
                                use_container_width=True)
    st.markdown("---")
    st.markdown("### 🖼️ تحلیل تصویر" if language == "فارسی" else "### 🖼️ Image analysis")
    uploaded_image = st.file_uploader("تصویر (PNG/JPG/WebP)" if language == "فارسی" else "Upload image (PNG/JPG/WebP)",
                                      type=["png", "jpg", "jpeg", "webp"], key="image_upload")
    use_web = st.toggle("جست‌وجوی آنلاین" if language == "فارسی" else "Online web search", value=True)
    st.caption("Google Custom Search / SerpAPI در صورت تنظیم کلید؛ در غیر این صورت DuckDuckGo." if language == "فارسی"
               else "Uses Google Custom Search / SerpAPI when configured; otherwise DuckDuckGo.")
    st.markdown("---")
    st.markdown("### 📤 خروجی" if language == "فارسی" else "### 📤 Export")
    st.caption("پاسخ‌ها و گفتگو را می‌توانید پس از تولید دانلود کنید." if language == "فارسی"
               else "Download the conversation after generating responses.")

title = "دستیار هوشمند PDF، وب و تصویر" if language == "فارسی" else "Smart PDF, Web & Image Assistant"
subtitle = "جست‌وجوی دقیق در چند سند، پاسخ مستند با شماره صفحه، آموزش مرحله‌به‌مرحله و آزمون‌سازی؛ با رابط تیره و نئونی." if language == "فارسی" else "Search across multiple documents, cite page numbers, learn step by step, create quizzes, and search the web."
st.markdown(f"""<div class="main-header"><h1>✦ {title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)

defaults = {
    "chat_history": [], "vector_store": None, "embeddings": None,
    "page_texts": {}, "processed_signature": None, "processed_names": [],
    "last_answer": "", "index_stats": {"pages": 0, "chunks": 0},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def normalize_digits(value: str) -> str:
    return value.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"))

def requested_page_number(query: str) -> Optional[int]:
    q = normalize_digits(query.casefold())
    for pattern in (r"(?:صفحه(?:ٔ|ی)?|page|p\.)\s*(\d+)", r"(\d+)\s*(?:صفحه|page)"):
        match = re.search(pattern, q)
        if match:
            return int(match.group(1))
    return None

def make_signature(files) -> tuple:
    return tuple(sorted((f.name, f.size) for f in (files or [])))

def google_search(query: str, limit: int = 6) -> str:
    """Use official Google Custom Search or SerpAPI if configured; DDGS is the fallback."""
    results = []
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            import requests
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": min(limit, 10)},
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("items", []):
                results.append({"title": item.get("title", ""), "url": item.get("link", ""),
                                "snippet": item.get("snippet", "")})
        except Exception:
            logger.exception("Google Custom Search failed")
    if not results and SERPAPI_API_KEY:
        try:
            import requests
            response = requests.get("https://serpapi.com/search.json",
                params={"engine": "google", "q": query, "api_key": SERPAPI_API_KEY, "num": limit},
                timeout=12)
            response.raise_for_status()
            for item in response.json().get("organic_results", [])[:limit]:
                results.append({"title": item.get("title", ""), "url": item.get("link", ""),
                                "snippet": item.get("snippet", "")})
        except Exception:
            logger.exception("SerpAPI Google search failed")
    if not results and DDGS is not None:
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=limit):
                    results.append({"title": item.get("title", ""), "url": item.get("href", ""),
                                    "snippet": item.get("body", "")})
        except Exception:
            logger.exception("DuckDuckGo search failed")
    if not results:
        return "No web results available. Configure GOOGLE_API_KEY + GOOGLE_CSE_ID or SERPAPI_API_KEY."
    return "\n\n".join(
        f"{i}. {x['title']}\nURL: {x['url']}\nSnippet: {x['snippet']}"
        for i, x in enumerate(results, 1)
    )

def build_history(history):
    msgs = []
    for item in history[-12:]:
        if item["role"] == "user":
            msgs.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            msgs.append(AIMessage(content=item["content"]))
    return msgs

def format_docs(docs):
    blocks = []
    seen = set()
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "PDF")
        key = (source, page, doc.page_content[:100])
        if key in seen:
            continue
        seen.add(key)
        blocks.append(f"[منبع PDF: {source}، صفحه {page}]\\n{doc.page_content}")
    return "\n\n".join(blocks)

if process_clicked:
    if not uploaded_pdfs:
        st.warning("ابتدا حداقل یک PDF انتخاب کنید." if language == "فارسی" else "Choose at least one PDF first.")
    else:
        try:
            with st.spinner("در حال استخراج متن و ساخت نمایه..." if language == "فارسی" else "Extracting text and building index..."):
                documents: List[Document] = []
                page_texts: Dict[str, Dict[int, str]] = {}
                names = []
                for pdf in uploaded_pdfs:
                    names.append(pdf.name)
                    reader = PdfReader(io.BytesIO(pdf.getvalue()))
                    page_texts[pdf.name] = {}
                    for page_no, page in enumerate(reader.pages, start=1):
                        text = (page.extract_text() or "").strip()
                        if text:
                            page_texts[pdf.name][page_no] = text
                            documents.append(Document(page_content=text, metadata={"page": page_no, "source": pdf.name}))
                if not documents:
                    st.error("متن قابل استخراجی پیدا نشد؛ PDF اسکن‌شده به OCR نیاز دارد." if language == "فارسی"
                             else "No extractable text found. Scanned PDFs need OCR.")
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=180,
                                                               add_start_index=True, separators=["\n\n", "\n", "۔", ". ", " ", ""])
                    chunks = splitter.split_documents(documents)
                    if st.session_state.embeddings is None:
                        st.session_state.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                            encode_kwargs={"normalize_embeddings": True},
                        )
                    st.session_state.vector_store = FAISS.from_documents(chunks, st.session_state.embeddings)
                    st.session_state.page_texts = page_texts
                    st.session_state.processed_signature = make_signature(uploaded_pdfs)
                    st.session_state.processed_names = names
                    st.session_state.chat_history = []
                    st.session_state.index_stats = {"pages": len(documents), "chunks": len(chunks)}
                    st.success(f"✅ {len(names)} فایل، {len(documents)} صفحه دارای متن و {len(chunks)} قطعه آماده شد." if language == "فارسی"
                               else f"✅ Indexed {len(names)} files, {len(documents)} text pages, and {len(chunks)} chunks.")
        except Exception as exc:
            logger.exception("PDF processing failed")
            st.error(f"خطا در پردازش PDF: {exc}" if language == "فارسی" else f"PDF processing error: {exc}")

if st.session_state.processed_names:
    c1, c2, c3 = st.columns(3)
    c1.metric("فایل‌ها" if language == "فارسی" else "PDF files", len(st.session_state.processed_names))
    c2.metric("صفحه‌های متنی" if language == "فارسی" else "Text pages", st.session_state.index_stats["pages"])
    c3.metric("قطعه‌های بازیابی" if language == "فارسی" else "Retrieval chunks", st.session_state.index_stats["chunks"])
    with st.expander("فایل‌های نمایه‌شده / Indexed files"):
        for name in st.session_state.processed_names:
            st.write("• " + name)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if language == "فارسی":
            st.markdown(f'<div dir="rtl" style="text-align:right;line-height:2">{html.escape(msg["content"]).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

def ask_model(query: str, image_bytes: Optional[bytes] = None, image_type: str = "image/png") -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY تنظیم نشده است. آن را در Secrets یا environment قرار دهید.")
    llm = ChatGroq(model=MODEL_NAME, groq_api_key=GROQ_API_KEY, temperature=0.2)
    pdf_context = "No PDF has been processed."
    store = st.session_state.vector_store
    page = requested_page_number(query)
    page_texts = st.session_state.page_texts
    if page is not None and page_texts:
        matches = []
        for filename, pages in page_texts.items():
            if page in pages:
                matches.append(f"[منبع PDF: {filename}، صفحه {page}]\n{pages[page]}")
        pdf_context = "\n\n".join(matches) if matches else (
            f"صفحه {page} با متن قابل استخراج پیدا نشد. فایل‌ها: {', '.join(page_texts.keys())}")
    elif store is not None:
        docs = store.similarity_search(query, k=7)
        pdf_context = format_docs(docs) if docs else "No relevant PDF passages found."
    web_context = google_search(query) if use_web else "Web search disabled by user."
    tutor_rules = ""
    if mode in ("معلم خصوصی", "Private Tutor"):
        tutor_rules = "Act as a patient private tutor. Explain concepts step by step, check understanding, use examples, and ask one short follow-up question."
    elif mode in ("ساخت آزمون", "Create Quiz"):
        tutor_rules = "Create a useful quiz based on the supplied PDF context first. Include 5 questions, mixed multiple-choice and short-answer, then put the answer key in a separate clearly labeled section at the end. Do not fabricate PDF facts."
    system = f"""You are NOVA, a precise PDF research assistant and private tutor. Reply in the user's language ({language}).
{tutor_rules}
Prioritize retrieved PDF evidence and cite every PDF claim as [نام فایل، صفحه N]. Never invent page numbers.
For web claims, attach numbered citations like [Web 1] and include the exact URL from the search results in a Sources section. Never invent URLs or claim a source supports more than its snippet supports.
Clearly distinguish document evidence from web information. If evidence is missing, say so. For calculations show steps.
PDF CONTEXT:
{pdf_context}

WEB SEARCH RESULTS:
{web_context}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    chain = ({"chat_history": lambda _: build_history(st.session_state.chat_history),
              "input": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    if image_bytes:
        # Groq vision models accept OpenAI-compatible image_url content blocks.
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{image_type};base64,{encoded}"
        vision_llm = ChatGroq(model=VISION_MODEL, groq_api_key=GROQ_API_KEY, temperature=0.2)
        message = HumanMessage(content=[
            {"type": "text", "text": query or ("این تصویر را توصیف و تحلیل کن." if language == "فارسی" else "Describe and analyze this image.")},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])
        response = vision_llm.invoke([("system", system)] + build_history(st.session_state.chat_history) + [message])
        return response.content if isinstance(response.content, str) else str(response.content)
    return chain.invoke(query)

user_query = st.chat_input("درباره PDF بپرسید، تصویر را تحلیل کنید یا موضوعی را جست‌وجو کنید..." if language == "فارسی"
                           else "Ask about your PDFs, analyze the image, or search a topic...")

if user_query:
    if uploaded_pdfs and st.session_state.processed_signature != make_signature(uploaded_pdfs):
        st.warning("فایل‌ها تغییر کرده‌اند؛ دوباره پردازش کنید." if language == "فارسی" else "Files changed; process them again.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            try:
                with st.spinner("در حال تحلیل و بازیابی منابع..." if language == "فارسی" else "Analyzing and retrieving sources..."):
                    image_bytes = uploaded_image.getvalue() if uploaded_image else None
                    image_type = uploaded_image.type if uploaded_image else "image/png"
                    answer = ask_model(user_query, image_bytes, image_type)
                if language == "فارسی":
                    st.markdown(f'<div dir="rtl" style="text-align:right;line-height:2;overflow-wrap:anywhere">{html.escape(answer).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.last_answer = answer
            except Exception as exc:
                logger.exception("Assistant request failed")
                st.error(f"درخواست ناموفق بود: {exc}" if language == "فارسی" else f"Request failed: {exc}")

if st.session_state.chat_history:
    conversation = []
    for item in st.session_state.chat_history:
        role = "شما" if language == "فارسی" and item["role"] == "user" else ("دستیار" if language == "فارسی" else item["role"].title())
        conversation.append(f"## {role}\n\n{item['content']}")
    export_md = "\n\n---\n\n".join(conversation)
    st.download_button("⬇️ دانلود گفتگو به Markdown" if language == "فارسی" else "⬇️ Download conversation as Markdown",
                       data=export_md, file_name="nova_conversation.md", mime="text/markdown")
    st.download_button("⬇️ دانلود گفتگو به TXT" if language == "فارسی" else "⬇️ Download conversation as TXT",
                       data=export_md, file_name="nova_conversation.txt", mime="text/plain")

with st.expander("⚙️ تنظیمات راه‌اندازی و کلیدها" if language == "فارسی" else "⚙️ Setup and API keys"):
    st.markdown("""
**کلیدهای اختیاری برای جست‌وجو**
- `GOOGLE_API_KEY` و `GOOGLE_CSE_ID`: برای Google Custom Search JSON API
- `SERPAPI_API_KEY`: جایگزین برای جست‌وجوی نتایج Google
- در نبود کلیدهای گوگل، جست‌وجو با DuckDuckGo انجام می‌شود.
- `GROQ_VISION_MODEL`: مدل بینایی قابل‌دسترسی در حساب Groq شما (پیش‌فرض `meta-llama/llama-4-scout-17b-16e-instruct`)
- برای OCR اسناد اسکن‌شده، در نسخه بعد می‌توان `pytesseract` و موتور Tesseract را اضافه کرد.
""" if language == "فارسی" else """
**Optional search keys**
- `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`: Google Custom Search JSON API
- `SERPAPI_API_KEY`: Google results via SerpAPI
- Without Google credentials, search falls back to DuckDuckGo.
- `GROQ_VISION_MODEL`: a vision model available to your Groq account (defaults to `meta-llama/llama-4-scout-17b-16e-instruct`).
- Scanned-PDF OCR can be added with `pytesseract` plus the Tesseract engine.
""")
