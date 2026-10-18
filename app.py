from __future__ import annotations

import io
import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NOVA AI Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# MULTILINGUAL DICTIONARIES
# =========================================================

TEXTS = {
    "🇮🇷 فارسی": {
        "title": "NOVA 2.0 · دستیار هوشمند",
        "subtitle": "هوش مصنوعی شخصی · مطالعه · وب · اسناد · خروجی",
        "sidebar_workspace": "فضای کاری",
        "sidebar_lang": "🌐 انتخاب زبان",
        "new_chat": "چت جدید",
        "system_status": "وضعیت سیستم",
        "online_model": "مدل آنلاین",
        "connected": "متصل به تنظیمات API",
        "not_configured": "تنظیم نشده",
        "web_search": "جست‌وجوی وب",
        "ready": "آماده",
        "need_install": "نیازمند نصب",
        "uploaded_docs": "اسناد بارگذاری‌شده",
        "messages_count": "پیام‌ها",
        "clear_chat": "پاک‌کردن تاریخچه گفتگو",
        "chat_page": "گفت‌وگو",
        "study_page": "مطالعه و تمرین",
        "docs_page": "مدیریت اسناد",
        "web_page": "جست‌وجوی وب",
        "export_page": "خروجی و پشتیبان",
        "settings_page": "تنظیمات",
        "chat_input": "پیام، سوال یا تمرین خود را برای NOVA بنویس...",
        "thinking": "NOVA در حال تحلیل دقیق و آماده‌سازی پاسخ است...",
        "error_msg": "در تولید پاسخ خطایی رخ داد. تنظیمات API و اتصال شبکه را بررسی کنید.",
        "upload_label": "افزودن فایل/عکس",
        "ask_doc_input": "سؤال درباره محتوای PDF یا تمرین‌ها",
        "ask_doc_btn": "پاسخ از روی اسناد و وب",
        "study_text": "موضوع یا متن آموزشی یا صورت تمرین",
        "generate_study": "تولید محتوای آموزشی",
        "flashcard_text": "متن برای ساخت فلش‌کارت",
        "make_flashcard": "ساخت فلش‌کارت",
    },
    "🇺🇸 English": {
        "title": "NOVA 2.0 · Smart Assistant",
        "subtitle": "Personal AI · Study · Web · Documents · Export",
        "sidebar_workspace": "Workspace",
        "sidebar_lang": "🌐 Select Language",
        "new_chat": "New Chat",
        "system_status": "System Status",
        "online_model": "Online Model",
        "connected": "Connected to API",
        "not_configured": "Not Configured",
        "web_search": "Web Search",
        "ready": "Ready",
        "need_install": "Needs Installation",
        "uploaded_docs": "Uploaded Documents",
        "messages_count": "Messages",
        "clear_chat": "Clear Chat History",
        "chat_page": "Chat",
        "study_page": "Study & Practice",
        "docs_page": "Documents",
        "web_page": "Web Research",
        "export_page": "Export & Backup",
        "settings_page": "Settings",
        "chat_input": "Type your message, question, or exercise for NOVA...",
        "thinking": "NOVA is analyzing and preparing the response...",
        "error_msg": "An error occurred. Check API settings and network connection.",
        "upload_label": "Add File/Image",
        "ask_doc_input": "Question about PDF content or exercises",
        "ask_doc_btn": "Answer from Docs & Web",
        "study_text": "Subject, educational text, or exercise prompt",
        "generate_study": "Generate Study Content",
        "flashcard_text": "Text for creating flashcards",
        "make_flashcard": "Create Flashcards",
    }
}


# =========================================================
# SESSION STATE
# =========================================================

SESSION_DEFAULTS = {
    "chat_history": [],
    "documents": {},
    "document_hashes": {},
    "active_mode": "General Assistant",
    "last_answer": "",
    "web_results": [],
    "quiz": [],
    "flashcards": [],
    "language": "🇮🇷 فارسی",
    "settings": {
        "model": "openai/gpt-oss-120b",
        "temperature": 0.1,
        "max_tokens": 2048,
    },
}

for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = (
            default.copy() if isinstance(default, dict)
            else default.copy() if isinstance(default, list)
            else default
        )


# =========================================================
# STYLING (Forcing text wrapping inside chat box and tables)
# =========================================================

current_lang = st.session_state.get("language", "🇮🇷 فارسی")
is_rtl = "فارسی" in current_lang or "עברית" in current_lang

st.markdown(
    f"""
    <style>
    .stApp {{
        background: #080b14;
        color: #e7edf7;
        font-family: Tahoma, sans-serif;
    }}

    [data-testid="stSidebar"] {{
        background: #0c1120;
        border-right: 1px solid #252e49;
    }}

    h1, h2, h3 {{
        color: #e8faff !important;
    }}

    .nova-header-title {{
        font-size: 1.5rem;
        font-weight: 800;
        color: #5cecff;
        margin-bottom: 0.1rem;
    }}

    .nova-header-sub {{
        color: #9baac5;
        font-size: 0.8rem;
        margin-bottom: 0.8rem;
    }}

    /* جلوگیری از بیرون زدن متن و کادر چت در موبایل */
    div[data-testid="stChatMessage"] {{
        border: 1px solid #27334e;
        border-radius: 12px;
        background: rgba(15, 22, 39, .95);
        padding: 14px;
        text-align: {"right" if is_rtl else "left"};
        direction: {"rtl" if is_rtl else "ltr"};
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
    }}

    .stButton button {{
        border: 1px solid #354b6d;
        border-radius: 8px;
        width: 100%;
    }}

    /* استایل استاندارد جدول‌ها داخل کادر پیام برای جلوگیری از بیرون زدگی */
    table {{
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 10px 0;
        background-color: #0f1627;
        border-radius: 8px;
        overflow: hidden;
        display: table !important;
    }}
    
    th, td {{
        padding: 8px 10px !important;
        border: 1px solid #27334e !important;
        text-align: center !important;
        font-size: 0.9rem;
        word-break: break-word;
    }}
    
    th {{
        background-color: #1a233a !important;
        color: #5cecff !important;
    }}
    
    .plus-btn-container button {{
        background: transparent !important;
        border: 1px solid #354b6d !important;
        color: #5cecff !important;
        font-size: 1.2rem !important;
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

APP_DIR = Path(__file__).resolve().parent
EXPORT_DIR = APP_DIR / "exports"

def t(key: str) -> str:
    lang = st.session_state.get("language", "🇮🇷 فارسی")
    if lang in TEXTS and key in TEXTS[lang]:
        return TEXTS[lang][key]
    return TEXTS["🇺🇸 English"].get(key, key)


def get_api_key() -> str:
    try:
        value = st.secrets.get("GROQ_API_KEY", "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()


def make_signature(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_pdf_text(data: bytes) -> list[dict]:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed.")
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append({"page": page_number, "text": text.strip()})
    return pages


def get_document_context() -> str:
    parts = []
    remaining = 28000
    for filename, pages in st.session_state.documents.items():
        if remaining <= 0:
            break
        for page in pages:
            if remaining <= 0:
                break
            section = f"\n[مدرک: {filename} | صفحه: {page['page']}]\n{page['text']}\n"
            section = section[:remaining]
            parts.append(section)
            remaining -= len(section)
    return "\n".join(parts)


def search_web(query: str, limit: int = 5) -> list[dict]:
    if DDGS is None:
        return []
    try:
        with DDGS() as search_client:
            results = search_client.text(query, max_results=limit)
        normalized = []
        for item in results or []:
            normalized.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("href", ""),
                "snippet": item.get("body", ""),
            })
        return normalized
    except Exception:
        return []


def generate_answer(question: str, mode: str, web_results: list[dict] | None = None) -> str:
    api_key = get_api_key()
    if Groq is None:
        return "Groq library is not installed."
    if not api_key:
        return "GROQ_API_KEY is not configured."

    settings = st.session_state.settings
    client = Groq(api_key=api_key)

    selected_language = st.session_state.get("language", "🇮🇷 فارسی")

    greetings = ["سلام", "درود", "hello", "hi", "hola", "ciao", "привет", "merhaba", "你好", "こんにちは", "שלום"]
    is_greeting = any(greet in question.lower() for greet in greetings) and len(question.split()) <= 4

    if is_greeting:
        system_prompt = f"""
تو دستیار هوشمند NOVA هستی. به گرمی و با ادب به زبان {selected_language} جواب بده و اعلام آمادگی کن.
"""
    else:
        # پرامپت بسیار دقیق، تضمینی، ساده برای کودکان 12 ساله، و اجبار کامل به پاسخ‌گویی مستقیم بدون امتناع
        system_prompt = f"""
تو هوش مصنوعی بسیار دقیق و حرفه‌ای به نام NOVA هستی. کاربر تو یک دانش‌آموز ۱۲ ساله است.

قوانین حیاتی و قطعی:
۱. **زبان پاسخ:** فقط و فقط به زبان {selected_language} پاسخ بده.
۲. **پاسخ مستقیم و تضمینی:** هرگز از پاسخ دادن به سوالات یا حل تمرین‌ها امتناع نکن، هیچ‌وقت پند و اندرز نده که "خودت حل کن"، و همیشه جواب کامل، دقیق و قطعی سوالات موجود در فایل آپلود شده یا متن را بده.
۳. **سادگی برای سن ۱۲ سال:** تمام مفاهیم، فرمول‌ها و پاسخ‌ها را مثل یک معلم مهربان و با زبان بسیار ساده، روان و کودکانه (مناسب سن ۱۲ سال) توضیح بده.
۴. **دقت کامل:** سوال را با نهایت دقت بررسی کن و پاسخ کاملاً صحیح، قدم به قدم و بدون اشتباه را بنویس. از نمادهای ریاضی استاندارد مثل ضرب (×) و تقسیم (÷) استفاده کن. به هیچ وجه از کدهای کامپیوتری استفاده نکن.
"""

    context = get_document_context()
    if context and not is_greeting:
        context = "\n\nمتن اسناد و فایل‌های بارگذاری‌شده:\n" + context

    if not is_greeting and web_results:
        context += "\n\nنتایج جستجوی وب:\n"
        for index, item in enumerate(web_results, start=1):
            context += f"\n[{index}] {item['title']}\nتوضیح: {item['snippet']}\n"

    messages = [{"role": "system", "content": system_prompt}]
    if context and not is_greeting:
        messages.append({"role": "system", "content": context})

    for message in st.session_state.chat_history[-12:]:
        if message.get("role") in ("user", "assistant"):
            messages.append({"role": message["role"], "content": message.get("content", "")})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=float(settings["temperature"]),
            max_tokens=int(settings["max_tokens"]),
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"خطا در ارتباط با هوش مصنوعی: {e}"


def download_name(prefix: str, extension: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def export_chat_markdown() -> str:
    lines = ["# NOVA Conversation", "", f"Exported: {datetime.now().isoformat(timespec='seconds')}", ""]
    for message in st.session_state.chat_history:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        lines.extend([f"## {role}", "", content, ""])
    return "\n".join(lines)


def export_chat_json() -> str:
    payload = {
        "application": "NOVA",
        "version": "2.0",
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.chat_history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("### NOVA 2.0")
    st.divider()

    if st.button(t("new_chat"), use_container_width=True, type="primary"):
        st.session_state.chat_history = []
        st.session_state.last_answer = ""
        st.rerun()

    st.divider()

    languages_list = [
        "🇮🇷 فارسی",
        "🇺🇸 English",
    ]
    
    selected_lang = st.selectbox(
        t("sidebar_lang"),
        languages_list,
        index=languages_list.index(st.session_state.language) if st.session_state.language in languages_list else 0
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

    st.divider()

    page = st.radio(
        t("sidebar_workspace"),
        [
            t("chat_page"),
            t("study_page"),
            t("docs_page"),
            t("web_page"),
            t("export_page"),
            t("settings_page"),
        ],
        key="navigation",
    )

    st.divider()
    st.markdown(f"### {t('system_status')}")

    api_available = bool(get_api_key()) and Groq is not None
    st.write(f"{t('online_model')}: " + (t("connected") if api_available else t("not_configured")))
    st.write(f"{t('web_search')}: " + (t("ready") if DDGS is not None else t("need_install")))
    st.write(f"{t('uploaded_docs')}: {len(st.session_state.documents)}")
    st.write(f"{t('messages_count')}: {len(st.session_state.chat_history)}")

    if st.button(t("clear_chat"), use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_answer = ""
        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(f'<div class="nova-header-title">{t("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="nova-header-sub">{t("subtitle")}</div>', unsafe_allow_html=True)


# =========================================================
# CHAT PAGE
# =========================================================

if page == t("chat_page"):
    st.subheader(t("chat_page"))

    mode = st.selectbox(
        "Response Mode",
        ["General Assistant", "Study Tutor", "Research", "Writing"],
        index=["General Assistant", "Study Tutor", "Research", "Writing"].index(st.session_state.active_mode),
    )
    st.session_state.active_mode = mode

    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role if role in ("user", "assistant") else "assistant"):
            st.markdown(content)

    col_plus, col_input = st.columns([1, 11])
    with col_plus:
        st.markdown('<div class="plus-btn-container">', unsafe_allow_html=True)
        uploaded_chat_file = st.file_uploader("+", type=["pdf", "png", "jpg", "jpeg", "txt"], accept_multiple_files=False, key="chat_file_upload", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_input:
        prompt = st.chat_input(t("chat_input"))

    if uploaded_chat_file:
        try:
            file_data = uploaded_chat_file.getvalue()
            file_name = uploaded_chat_file.name
            if file_name.endswith(".pdf"):
                signature = make_signature(file_data)
                if st.session_state.document_hashes.get(file_name) != signature:
                    pages = extract_pdf_text(file_data)
                    st.session_state.documents[file_name] = pages
                    st.session_state.document_hashes[file_name] = signature
                    st.success(f"فایل PDF با موفقیت اضافه شد: {file_name}")
            elif file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                st.image(file_data, caption=file_name, width=250)
                st.success(f"تصویر بارگذاری شد: {file_name}")
        except Exception as e:
            st.error(f"خطا در پردازش فایل: {e}")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner(t("thinking")):
                    web_results = []
                    greetings = ["سلام", "درود", "hello", "hi", "hola", "ciao", "привет", "merhaba", "你好", "こんにちは", "שלום"]
                    is_greeting = any(g in prompt.lower() for g in greetings) and len(prompt.split()) <= 4
                    
                    if not is_greeting:
                        web_results = search_web(prompt, limit=3)

                    answer = generate_answer(prompt, mode, web_results=web_results)
                st.markdown(answer)
            except Exception as exc:
                answer = f"{t('error_msg')}\n\nDetails: `{exc}`"
                st.error(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "timestamp": datetime.now().isoformat()})
        st.session_state.last_answer = answer
        st.rerun()


# =========================================================
# DOCUMENTS PAGE
# =========================================================

elif page == t("docs_page"):
    st.subheader(t("docs_page"))

    uploaded_files = st.file_uploader(t("upload_label"), type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                data = uploaded_file.getvalue()
                signature = make_signature(data)
                filename = uploaded_file.name

                if st.session_state.document_hashes.get(filename) == signature:
                    continue

                with st.spinner(f"Processing {filename}..."):
                    pages = extract_pdf_text(data)

                st.session_state.documents[filename] = pages
                st.session_state.document_hashes[filename] = signature
                st.success(f"{filename}: {len(pages)} pages extracted.")
            except Exception as exc:
                st.error(f"Error {uploaded_file.name}: {exc}")

    st.divider()
    if st.session_state.documents:
        for filename, pages in list(st.session_state.documents.items()):
            with st.expander(f"{filename} · {len(pages)} pages"):
                if st.button(f"Delete {filename}", key=f"del_{filename}"):
                    del st.session_state.documents[filename]
                    st.session_state.document_hashes.pop(filename, None)
                    st.rerun()

    st.divider()
    document_question = st.text_input(t("ask_doc_input"))

    if st.button(t("ask_doc_btn")):
        if not st.session_state.documents:
            st.warning("Please upload a PDF first.")
        elif not document_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner(t("thinking")):
                    web_results = search_web(document_question, limit=3)
                    answer = generate_answer(document_question, "Research", web_results=web_results)
                st.markdown(answer)
                st.session_state.chat_history.extend([
                    {"role": "user", "content": document_question, "timestamp": datetime.now().isoformat()},
                    {"role": "assistant", "content": answer, "timestamp": datetime.now().isoformat()}
                ])
            except Exception as exc:
                st.error(f"Error: {exc}")


# =========================================================
# WEB RESEARCH PAGE
# =========================================================

elif page == t("web_page"):
    st.subheader(t("web_page"))
    query = st.text_input("Search Query")
    if st.button("Search", type="primary"):
        if query.strip():
            results = search_web(query, limit=5)
            st.session_state.web_results = results
            for item in results:
                st.markdown(f"### {item['title']}")
                st.write(item["snippet"])
                st.markdown(f"[Link]({item['url']})")


# =========================================================
# STUDY PAGE
# =========================================================

elif page == t("study_page"):
    st.subheader(t("study_page"))
    study_text = st.text_area(t("study_text"), height=180)
    if st.button(t("generate_study"), type="primary"):
        if study_text.strip():
            answer = generate_answer("Solve this exercise step by step: " + study_text, "Study Tutor", web_results=search_web(study_text, limit=3))
            st.markdown(answer)


# =========================================================
# EXPORT & STORAGE PAGE
# =========================================================

elif page == t("export_page"):
    st.subheader(t("export_page"))
    st.download_button("Download Markdown", data=export_chat_markdown(), file_name=download_name("nova", "md"), mime="text/markdown", use_container_width=True)
    st.download_button("Download JSON", data=export_chat_json(), file_name=download_name("nova", "json"), mime="application/json", use_container_width=True)


# =========================================================
# SETTINGS PAGE
# =========================================================

elif page == t("settings_page"):
    st.subheader(t("settings_page"))
    model = st.selectbox("Model", ["openai/gpt-oss-120b", "llama-3.1-70b-versatile"], index=0)
    if st.button("Save Settings"):
        st.session_state.settings["model"] = model
        st.success("Settings saved.")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("NOVA 2.0 · Multilingual AI Assistant")