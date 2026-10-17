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

# ۲. تنظیمات صفحه
st.set_page_config(
    page_title="Smart PDF Assistant | دستیار هوشمند PDF", 
    page_icon="🌿", 
    layout="wide"
)

# ۳. مدیریت سایدبار برای زبان و حالت تاریک/روشن
with st.sidebar:
    st.markdown("### 🌐 Language / انتخاب زبان")
    language = st.selectbox("", ["English", "فارسی"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### 🌗 Theme Mode / حالت نمایش")
    theme_mode = st.radio("", ["Dark Mode (کیهانی)", "Light Mode (جنگلی و طبیعی)"], label_visibility="collapsed")

# تنظیمات استایل داینامیک بر اساس حالت تاریک (کیهانی) یا روشن (جنگلی با متن کاملاً مشکی)
if "Dark" in theme_mode:
    bg_style = """
    .stApp {
        background-image: linear-gradient(rgba(10, 15, 30, 0.88), rgba(15, 23, 42, 0.94)), 
                          url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-header, .stChatMessage {
        background: rgba(30, 41, 59, 0.85) !important;
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #f8fafc !important;
    }
    .main-header h1, .main-header p, .stChatMessage p, .stChatMessage span, .stChatMessage div {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    """
else:
    bg_style = """
    .stApp {
        background-image: linear-gradient(rgba(240, 247, 240, 0.82), rgba(220, 240, 220, 0.88)), 
                          url('https://images.unsplash.com/photo-1448375240586-882707db888b?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-header, .stChatMessage {
        background: rgba(255, 255, 255, 0.92) !important;
        backdrop-filter: blur(14px);
        border: 1px solid rgba(0, 0, 0, 0.12);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        color: #0f172a !important;
    }
    .main-header h1, .main-header p, .stChatMessage p, .stChatMessage span, .stChatMessage div, label, span {
        color: #0f172a !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] {
        background-color: #f1f8f1;
        border-right: 1px solid #d1e7d1;
    }
    """

# اعمال استایل‌های CSS نهایی
st.markdown(f"""
    <style>
    {bg_style}
    .main-header {{
        padding: 2.2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
        border-right: 6px solid #10b981;
    }}
    .stChatMessage {{
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 12px;
    }}
    .stButton>button {{
        border-radius: 12px;
        font-weight: 600;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(16, 185, 129, 0.4);
    }}
    </style>
""", unsafe_allow_html=True)

# ۴. متن‌های دوزبانه
texts = {
    "English": {
        "title": "✨ Smart PDF Chat Assistant 🌿",
        "subtitle": "Upload your PDF file and get precise AI-powered answers instantly in both English and Persian.",
        "sidebar_header": "📄 Document Management",
        "upload_label": "Choose a PDF file",
        "process_btn": "🚀 Process & Analyze",
        "processing_spinner": "Extracting text and preparing AI...",
        "success_msg": "✅ File processed successfully! Ask your question now.",
        "chat_input": "Type your question here...",
        "warning_no_file": "⚠️ Please upload and process a PDF file first.",
        "searching_spinner": "Searching and generating smart response...",
    },
    "فارسی": {
        "title": "✨ دستیار هوشمند گفتگو با فایل‌های PDF 🌿",
        "subtitle": "فایل PDF خود را آپلود کنید و با استفاده از هوش مصنوعی پیشرفته، پاسخ دقیق سوالات خود را به هر دو زبان دریافت کنید.",
        "sidebar_header": "📄 مدیریت سند",
        "upload_label": "یک فایل PDF انتخاب کنید",
        "process_btn": "🚀 پردازش و تحلیل فایل",
        "processing_spinner": "در حال استخراج متن و آماده‌سازی هوش مصنوعی...",
        "success_msg": "✅ فایل با موفقیت پردازش شد! اکنون سوالتان را بپرسید.",
        "chat_input": "سوال خود را اینجا بنویسید...",
        "warning_no_file": "⚠️ لطفاً ابتدا یک فایل PDF آپلود و پردازش کنید.",
        "searching_spinner": "در حال جستجو و تولید پاسخ مدرن...",
    }
}

t = texts[language]

# نمایش هدر شیک
st.markdown(f"""
    <div class="main-header">
        <h1 style="margin-bottom: 8px; font-size: 26px;">{t['title']}</h1>
        <p style="font-size: 15px; margin: 0; line-height: 1.5;">{t['subtitle']}</p>
    </div>
""", unsafe_allow_html=True)

# ۵. مدیریت نشست (Session State)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ۶. بخش سایدبار برای آپلود فایل
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

# ۷. نمایش تاریخچه چت
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ۸. ورودی پیام کاربر و پردازش هوش مصنوعی
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
                You are a helpful AI assistant. Answer the user's question accurately based ONLY on the provided context. 
                - If the user asks in Persian, reply in fluent Persian. 
                - If the user asks in English, reply in fluent English.
                - If the answer is not in the text, say "در فایل یافت نشد / Not found in the file".
                
                Context:
                {context}
                
                Question: {input}
                """)

                def format_docs(docs):return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(user_query)

                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})