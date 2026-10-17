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

# ۲. تنظیمات صفحه با طراحی مدرن و UI جذاب
st.set_page_config(
    page_title="Smart PDF Assistant | دستیار هوشمند PDF", 
    page_icon="✨", 
    layout="wide"
)

# تزریق کدهای CSS برای زیباتر کردن ظاهر سایت، حاشیه‌ها و فونت‌ها
st.markdown("""
    <style>
    /* استایل کلی بدنه */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* بلوک‌های عنوان و هدر */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
        border-right: 6px solid #4f46e5;
    }
    
    /* استایل حباب‌های چت */
    .stChatMessage {
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        background-color: white;
    }
    
    /* دکمه‌ها و سایدبار */
    .css-1d391kg {
        background-color: #ffffff;
    }
    
    /* سفارشی‌سازی دکمه پردازش */
    .stButton>button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ۳. مدیریت دوزبانه بودن (فارسی و انگلیسی)
with st.sidebar:
    st.markdown("### 🌐 Language / انتخاب زبان")
    language = st.selectbox("", ["فارسی", "English"], label_visibility="collapsed")

# متن‌های ترجمه شده برای هر دو زبان
texts = {
    "فارسی": {
        "title": "✨ دستیار هوشمند گفتگو با اسناد PDF",
        "subtitle": "فایل PDF خود را آپلود کنید و با استفاده از هوش مصنوعی پیشرفته، پاسخ دقیق سوالات خود را به هر دو زبان دریافت کنید.",
        "sidebar_header": "📄 مدیریت سند",
        "upload_label": "یک فایل PDF انتخاب کنید",
        "process_btn": "🚀 پردازش و تحلیل فایل",
        "processing_spinner": "در حال استخراج متن و آماده‌سازی هوش مصنوعی...",
        "success_msg": "✅ فایل با موفقیت پردازش شد! اکنون سوالتان را بپرسید.",
        "chat_input": "سوال خود را اینجا بنویسید...",
        "warning_no_file": "⚠️ لطفاً ابتدا یک فایل PDF آپلود و پردازش کنید.",
        "searching_spinner": "در حال جستجو و تولید پاسخ مدرن...",
    },
    "English": {
        "title": "✨ Smart PDF Chat Assistant",
        "subtitle": "Upload your PDF file and get precise AI-powered answers instantly in both English and Persian.",
        "sidebar_header": "📄 Document Management",
        "upload_label": "Choose a PDF file",
        "process_btn": "🚀 Process & Analyze",
        "processing_spinner": "Extracting text and preparing AI...",
        "success_msg": "✅ File processed successfully! Ask your question now.",
        "chat_input": "Type your question here...",
        "warning_no_file": "⚠️ Please upload and process a PDF file first.",
        "searching_spinner": "Searching and generating smart response...",
    }
}

t = texts[language]

# نمایش هدر شیک و مدرن در بالای صفحه
st.markdown(f"""
    <div class="main-header">
        <h1 style="color: #1e293b; margin-bottom: 5px; font-size: 28px;">{t['title']}</h1>
        <p style="color: #64748b; font-size: 16px; margin: 0;">{t['subtitle']}</p>
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
                You are an advanced helpful AI assistant. Answer the user's question accurately based ONLY on the provided context. 
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