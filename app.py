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

# ۲. تنظیمات صفحه با طراحی مدرن و عنوان استاندارد
st.set_page_config(
    page_title="Smart PDF Assistant | دستیار هوشمند PDF", 
    page_icon="📚", 
    layout="wide"
)

# ۳. تزریق استایل‌های CSS سفارشی (پس‌زمینه کیهانی باکیفیت و طراحی شیشه‌ای مدرن)
st.markdown("""
    <style>
    /* تنظیم پس‌زمینه کیهانی و کوانتومی تاریک و شیک */
    .stApp {
        background-image: linear-gradient(rgba(11, 15, 25, 0.9), rgba(15, 23, 42, 0.95)), 
                          url('https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1920&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* هدر شیشه‌ای و مدرن با حاشیه گرادینت */
    .main-header {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 2.2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-right: 6px solid #6366f1;
    }
    
    /* استایل حباب‌های چت شیشه‌ای و خوانا */
    .stChatMessage {
        border-radius: 16px;
        padding: 1.2rem;
        background: rgba(30, 41, 59, 0.75) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    
    .stChatMessage p, .stChatMessage span, .stChatMessage div {
        color: #f8fafc !important;
        font-family: inherit;
    }
    
    /* منوی سایدبار شیشه‌ای */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.92);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* دکمه‌های جذاب با افکت هاور */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(124, 58, 237, 0.5);
    }
    
    /* جعبه متن چت پایین صفحه */
    .stChatInput input {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ۴. مدیریت دوزبانه بودن (فارسی و انگلیسی)
with st.sidebar:
    st.markdown("### 🌐 Language / انتخاب زبان")
    language = st.selectbox("", ["فارسی", "English"], label_visibility="collapsed")

# متن‌های ترجمه شده برای هر دو زبان
texts = {
    "فارسی": {
        "title": "📚 دستیار هوشمند گفتگو با فایل‌های PDF",
        "subtitle": "فایل PDF خود را آپلود کنید و با استفاده از هوش مصنوعی، پاسخ دقیق سوالات خود را به هر دو زبان دریافت کنید.",
        "sidebar_header": "📄 مدیریت سند",
        "upload_label": "یک فایل PDF انتخاب کنید",
        "process_btn": "🚀 پردازش و تحلیل فایل",
        "processing_spinner": "در حال استخراج متن و آماده‌سازی هوش مصنوعی...",
        "success_msg": "✅ فایل با موفقیت پردازش شد! اکنون سوالتان را بپرسید.",
        "chat_input": "سوال خود را درباره محتوای فایل بپرسید...",
        "warning_no_file": "⚠️ لطفاً ابتدا یک فایل PDF آپلود و پردازش کنید.",
        "searching_spinner": "در حال جستجو در سند و تولید پاسخ...",
    },
    "English": {
        "title": "📚 Smart PDF Chat Assistant",
        "subtitle": "Upload your PDF file and intelligently chat with it in English or Persian!",
        "sidebar_header": "📄 Document Management",
        "upload_label": "Choose a PDF file",
        "process_btn": "🚀 Process & Analyze File",
        "processing_spinner": "Extracting text and processing intelligently...",
        "success_msg": "✅ File processed successfully! You can now ask questions.",
        "chat_input": "Ask a question about the file content...",
        "warning_no_file": "⚠️ Please upload and process a PDF file first.",
        "searching_spinner": "Searching document and generating answer...",
    }
}

t = texts[language]

# نمایش هدر شیک و مدرن
st.markdown(f"""
    <div class="main-header">
        <h1 style="color: #f8fafc; margin-bottom: 8px; font-size: 27px; font-weight: 700;">{t['title']}</h1>
        <p style="color: #94a3b8; font-size: 15px; margin: 0; line-height: 1.5;">{t['subtitle']}</p>
    </div>
""", unsafe_allow_html=True)

# ۵. مدیریت حافظه نشست (Session State)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ۶. منوی سایدبار برای آپلود فایل
with st.sidebar:
    st.markdown("---")
    st.markdown(f"### {t['sidebar_header']}")
    uploaded_