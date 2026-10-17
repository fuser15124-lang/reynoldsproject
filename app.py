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

# ۱. بارگذاری متغیرها و کلید API
load_dotenv()

# ۲. تنظیمات صفحه و ظاهر مدرن‌تر (UI/UX)
st.set_page_config(
    page_title="Smart PDF Assistant | دستیار هوشمند PDF", 
    page_icon="📚", 
    layout="wide"
)

# استایل‌دهی سفارشی برای ظاهر زیباتر
st.markdown("""
    <style>
    .stApp {
        background-color: #fafbfc;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ۳. مدیریت دوزبانه بودن برنامه‌ها (فارسی و انگلیسی)
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/pdf.png", width=80)
    st.markdown("### 🌐 Language / انتخاب زبان")
    language = st.selectbox("", ["فارسی", "English"], label_visibility="collapsed")

# متن‌های ترجمه شده برای هر دو زبان
texts = {
    "فارسی": {
        "title": "📚 دستیار هوشمند گفتگو با فایل‌های PDF",
        "subtitle": "فایل PDF خود را آپلود کنید و به‌صورت هوشمند به دو زبان فارسی یا انگلیسی پاسخ بگیرید!",
        "sidebar_header": "📄 مدیریت سند",
        "upload_label": "یک فایل PDF انتخاب کنید",
        "process_btn": "🚀 پردازش و تحلیل فایل",
        "processing_spinner": "در حال استخراج متن و پردازش هوشمند...",
        "success_msg": "✅ فایل با موفقیت پردازش شد! حالا سوالت رو بپرس.",
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

# نمایش عنوان و توضیحات
st.title(t["title"])
st.markdown(f"*{t['subtitle']}*")
st.divider()

# ۴. مدیریت حافظه نشست (Session State)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ۵. منوی سایدبار برای آپلود فایل
with st.sidebar:
    st.markdown("---")
    st.header(t["sidebar_header"])
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

                # پرامپت هوشمند برای پشتیبانی از هر دو زبان
                prompt = ChatPromptTemplate.from_template("""
                You are a helpful AI assistant. Answer the user's question accurately based ONLY on the provided context. 
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