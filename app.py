import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ۱. بارگذاری متغیرها
load_dotenv()

# ۲. تنظیمات صفحه با طراحی کیهانی و مدرن
st.set_page_config(
    page_title="Quantum & Cosmos PDF Assistant | دستیار کوانتومی PDF", 
    page_icon="🌌", 
    layout="wide"
)

# تزریق استایل‌های CSS سفارشی و تصویر پس‌زمینه کیهانی
st.markdown("""
    <style>
    /* تنظیم تصویر پس‌زمینه سه‌بعدی کیهان و فیزیک کوانتوم */
    .stApp {
        background-image: linear-gradient(rgba(10, 15, 30, 0.85), rgba(15, 23, 42, 0.9), rgba(10, 15, 30, 0.95)), 
                          url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* بلوک هدر شیشه‌ای و مدرن */
    .main-header {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-right: 6px solid #8b5cf6;
    }
    
    /* استایل حباب‌های چت شیشه‌ای */
    .stChatMessage {
        border-radius: 14px;
        padding: 1rem;
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* استایل متن‌های داخل چت برای خوانایی بهتر در پس‌زمینه تاریک */
    .stChatMessage p, .stChatMessage span, .stChatMessage div {
        color: #f1f5f9 !important;
    }
    
    /* منوی سایدبار شیشه‌ای */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* دکمه‌های جذاب */
    .stButton>button {
        border-radius: 10px;
        font-weight: bold;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ۳. مدیریت دوزبانه بودن (فارسی و انگلیسی)
with st.sidebar:
    st.markdown("### 🌌 Language / انتخاب زبان")
    language = st.selectbox("", ["فارسی", "English"], label_visibility="collapsed")

# متن‌های ترجمه شده برای هر دو زبان
texts = {
    "فارسی": {
        "title": "🌌 دستیار هوشمند کوانتوم و کیهان‌شناسی",
        "subtitle": "فایل PDF خود را بارگذاری کنید و از طریق هوش مصنوعی پیشرفته، پاسخ پرسش‌های خود را دریافت کنید.",
        "sidebar_header": "📄 مدیریت اسناد",
        "upload_label": "انتخاب فایل PDF",
        "process_btn": "🚀 پردازش و تحلیل سند",
        "processing_spinner": "در حال کاوش در ساختار متن...",
        "success_msg": "✅ فایل با موفقیت پردازش شد! اکنون سوالتان را بپرسید.",
        "chat_input": "پرسش خود را درباره کیهان، کوانتوم یا متن فایل بنویسید...",
        "warning_no_file": "⚠️ لطفاً ابتدا یک فایل PDF آپلود و پردازش کنید.",
        "searching_spinner": "در حال کاوش در کهکشان داده‌ها...",
    },
    "English": {
        "title": "🌌 Quantum & Cosmos PDF Assistant",
        "subtitle": "Upload your PDF file and explore precise AI-powered answers instantly.",
        "sidebar_header": "📄 Document Management",
        "upload_label": "Choose a PDF file",
        "process_btn": "🚀 Process & Analyze",
        "processing_spinner": "Extracting and analyzing text structure...",
        "success_msg": "✅ File processed successfully! Ask your question now.",
        "chat_input": "Ask about quantum physics, cosmos, or file content...",
        "warning_no_file": "⚠️ Please upload and process a PDF file first.",
        "searching_spinner": "Exploring the galaxy of data...",
    }
}

t = texts[language]

# نمایش هدر شیک و کیهانی
st.markdown(f"""
    <div class="main-header">
        <h1 style="color: #f8fafc; margin-bottom: 5px; font-size: 26px;">{t['title']}</h1>
        <p style="color: #cbd5e1; font-size: 15px; margin: 0;">{t['subtitle']}</p>
    </div>
""", unsafe_allow_html=True)

# ۴. مدیریت حافظه نشست (Session State)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ۵. منوی سایدبار برای آپلود فایل
with st.sidebar:
    st.markdown("---")
    st.markdown(f"### {t['sidebar_header']}")
    uploaded_file = st.file_uploader(t["upload_label"], type=["pdf"])
    
    if uploaded_file is not None and st.button(t["process_btn"], use_container_width=True):
        with st.spinner(t["processing_spinner"]):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            
            st.session_state.vector_store = vector_store
            st.success(t["success_msg"])

# ۶. نمایش تاریخچه گفتگو
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ۷. دریافت ورودی از کاربر
user_query = st.chat_input(t["chat_input"])

if user_query:
    if st.session_state.vector_store is None:
        st.warning(t["warning_no_file"])
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner(t["searching_spinner"]):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)

                prompt = ChatPromptTemplate.from_template("""
                You are an advanced AI assistant specialized in physics, cosmos, and document analysis. Answer the user's question accurately based ONLY on the provided context. 
                - If the user asks in Persian, reply in fluent Persian. 
                - If the user asks in English, reply in fluent English.
                - If the answer is not in the text, say "در فایل یافت نشد / Not found in the file".
                
                Context:
                {context}
                
                Question: {input}
                """)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(user_query)

                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})