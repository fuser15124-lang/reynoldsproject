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

# ۱. بارگذاری متغیرها و کلید API از فایل .env
load_dotenv()

# ۲. تنظیمات اولیه صفحه Streamlit
st.set_page_config(page_title="دستیار هوشمند PDF", page_icon="📚", layout="wide")

st.title("📚 دستیار هوشمند گفتگو با فایل‌های PDF")
st.write("فایل PDF خود را در منوی سمت چپ آپلود کنید و هر سوالی دارید از آن بپرسید!")

# ۳. مدیریت حافظه نشست (Session State) برای ذخیره چت‌ها و دیتابیس
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ۴. منوی سمت چپ برای آپلود فایل PDF
with st.sidebar:
    st.header("📄 آپلود سند")
    uploaded_file = st.file_uploader("یک فایل PDF انتخاب کنید", type=["pdf"])
    
    if uploaded_file is not None and st.button("پردازش و تحلیل فایل"):
        with st.spinner("در حال استخراج متن و پردازش..."):
            # استخراج متن از تمامی صفحات PDF با کتابخانه pypdf
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            
            # خرد کردن متن به تکه‌های کوچک‌تر (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)
            
            # تبدیل تکه‌ها به بردارهای ریاضی با مدل HuggingFace و ذخیره در FAISS
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            
            st.session_state.vector_store = vector_store
            st.success("✅ فایل با موفقیت پردازش شد! اکنون می‌توانید سوال بپرسید.")

# ۵. نمایش پیام‌های قبلی چت
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ۶. دریافت سوال از کاربر
user_query = st.chat_input("سوال خود را درباره محتوای فایل بنویسید...")

if user_query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ لطفاً ابتدا یک فایل PDF آپلود کرده و دکمه پردازش را بزنید.")
    else:
        # نمایش سوال کاربر در صفحه
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # ساخت پاسخ هوشمند با استفاده از مدل LLaMA 3 در Groq
        with st.chat_message("assistant"):
            with st.spinner("در حال جستجو و پاسخ‌دهی..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                # استفاده از مدل پرسرعت LLaMA 3
                llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3)

                # طراحی دستور (Prompt) برای هوش مصنوعی
                prompt = ChatPromptTemplate.from_template("""
                پاسخ سوال کاربر را دقیقاً بر اساس اطلاعات موجود در متن زیر بده. اگر پاسخ در متن وجود نداشت، بگو در فایل یافت نشد.
                
                متن فایل:
                {context}
                
                سوال: {input}
                """)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                # ساخت زنجیره مدرن RAG بدون وابستگی به ماژول‌های قدیمی
                rag_chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                answer = rag_chain.invoke(user_query)

                # نمایش پاسخ و ذخیره در تاریخچه
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                 
