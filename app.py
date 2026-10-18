from __future__ import annotations

import io
import json
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st

# Optional dependencies are handled gracefully.
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
    page_title="NOVA AI",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE -- MUST RUN BEFORE ANY ACCESS
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
    "settings": {
        # اصلاح نام مدل به گزینه‌ای که در فایل سکرتس شما (openai/gpt-oss-120b) یا مدل‌های پایدار گروک است
        "model": "openai/gpt-oss-120b",
        "temperature": 0.3,
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
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at top left,
                rgba(37, 60, 105, .28), transparent 45%),
            #080b14;
        color: #e7edf7;
    }

    [data-testid="stSidebar"] {
        background: #0c1120;
        border-right: 1px solid #252e49;
    }

    h1, h2, h3 {
        color: #e8faff !important;
    }

    .nova-brand {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: .16em;
        color: #5cecff;
        text-shadow: 0 0 22px rgba(92, 236, 255, .3);
    }

    .nova-subtitle {
        color: #9baac5;
        font-size: .95rem;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid #27334e;
        border-radius: 14px;
        background: rgba(15, 22, 39, .72);
        padding: 12px;
    }

    .stButton button {
        border: 1px solid #354b6d;
        border-radius: 10px;
        transition: all .2s ease;
    }

    .stButton button:hover {
        border-color: #5cecff;
        color: #5cecff;
    }

    div[data-testid="stMetric"] {
        background: #10182a;
        padding: 14px;
        border: 1px solid #26334e;
        border-radius: 12px;
    }

    .nova-note {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #28415a;
        background: #0f1928;
        color: #b7c9df;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONFIGURATION AND HELPERS
# =========================================================

APP_DIR = Path(__file__).resolve().parent
EXPORT_DIR = APP_DIR / "exports"


def get_api_key() -> str:
    """Read the API key from Streamlit secrets or environment."""
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
    """Extract text page by page while preserving page numbers."""
    if PdfReader is None:
        raise RuntimeError(
            "کتابخانه pypdf نصب نیست. "
            "دستور pip install pypdf را اجرا کنید."
        )

    reader = PdfReader(io.BytesIO(data))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip(),
            })

    return pages


def get_document_context() -> str:
    """Build a bounded context from uploaded PDF files."""
    parts = []
    remaining = 24000

    for filename, pages in st.session_state.documents.items():
        if remaining <= 0:
            break

        for page in pages:
            if remaining <= 0:
                break

            section = (
                f"\n[Document: {filename} | "
                f"Page: {page['page']}]\n"
                f"{page['text']}\n"
            )

            section = section[:remaining]
            parts.append(section)
            remaining -= len(section)

    return "\n".join(parts)


def search_web(query: str, limit: int = 5) -> list[dict]:
    """Search the web if the optional provider is installed."""
    if DDGS is None:
        return []

    try:
        with DDGS() as search_client:
            results = search_client.text(
                query,
                max_results=limit,
            )

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


def generate_answer(
    question: str,
    mode: str,
    web_results: list[dict] | None = None,
) -> str:
    """Generate an answer using Groq, when configured."""
    api_key = get_api_key()

    if Groq is None:
        return (
            "کتابخانه Groq نصب نشده است. "
            "برای پاسخ‌گویی هوشمند، requirements.txt را نصب کنید."
        )

    if not api_key:
        return (
            "کلید API تنظیم نشده است.\n\n"
            "کلید GROQ_API_KEY را در فایل "
            ".streamlit/secrets.toml یا متغیر محیطی قرار دهید."
        )

    settings = st.session_state.settings
    client = Groq(api_key=api_key)

    # بررسی احوالپرسی یا مکالمه ساده برای اینکه بیهوده به دنبال PDF نگشته و مودبانه جواب دهد
    greetings = ["سلام", "درود", "خوب هستی", "چطور", "حالت چطوره", "hello", "hi", "hey", "ممنون", "تشکر"]
    is_greeting = any(greet in question.lower() for greet in greetings) and len(question.split()) <= 4

    if is_greeting:
        system_prompt = """
You are NOVA, a friendly, warm, and helpful AI assistant.
Respond warmly, politely, and naturally in the same language used by the user (Persian or English).
Offer your assistance with their PDF documents, exercises, questions, or general queries.
"""
    else:
        system_prompt = """
You are NOVA, an advanced expert AI academic tutor, researcher, and multilingual assistant.

Rules:
- Respond in the language used by the user.
- Distinguish verified evidence from uncertainty.
- Never invent citations, URLs, quotations, or page numbers.
- **First Priority (PDF Context):** If document context is supplied, prioritize it for document/study questions. Cite exact evidence and page numbers using [Document: filename, Page: N] or [صفحه N].
- **Second Priority (Web Search Context / Exercises):** If a question is an exercise or its answer is missing from the document context, use the provided web search results to give a fully correct, well-structured, and accurate answer. Mention if it comes from web sources.
- Clearly mention when the supplied context is insufficient.
- For study tasks and exercises, explain concepts step by step.
"""

        if mode == "Study Tutor":
            system_prompt += "\nAct as a study tutor. Explain reasoning, give examples, and provide step-by-step solutions to exercises."
        elif mode == "Research":
            system_prompt += "\nAct as a research assistant. Compare evidence and separate facts and interpretations."
        elif mode == "Writing":
            system_prompt += "\nAct as a writing assistant. Produce clear, structured text."

    context = get_document_context()

    if context and not is_greeting:
        context = (
            "\n\nUPLOADED DOCUMENT EXCERPTS:\n"
            + context
        )

    # اگر پاسخ در اسناد پیدا نشد یا سوال تمرین/خارج از جزوه بود، به‌صورت خودکار وب‌سرچ انجام شود
    if not is_greeting and web_results:
        context += "\n\nWEB SEARCH RESULTS (for exercises / missing info):\n"
        for index, item in enumerate(web_results, start=1):
            context += (
                f"\n[{index}] {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Excerpt: {item['snippet']}\n"
            )

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if context and not is_greeting:
        messages.append({
            "role": "system",
            "content": context,
        })

    for message in st.session_state.chat_history[-12:]:
        if message.get("role") in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message.get("content", ""),
            })

    messages.append({
        "role": "user",
        "content": question,
    })

    try:
        response = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=float(settings["temperature"]),
            max_tokens=int(settings["max_tokens"]),
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"خطا در ارتباط با مدل هوش مصنوعی: {e}"


def download_name(prefix: str, extension: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def export_chat_markdown() -> str:
    lines = [
        "# NOVA Conversation",
        "",
        f"Exported: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for message in st.session_state.chat_history:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        lines.extend([
            f"## {role}",
            "",
            content,
            "",
        ])

    return "\n".join(lines)


def export_chat_json() -> str:
    payload = {
        "application": "NOVA",
        "version": "2.0",
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.chat_history,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def build_flashcards(text: str) -> list[dict]:
    lines = [
        line.strip(" -*\t")
        for line in text.splitlines()
        if len(line.strip()) >= 20
    ]

    cards = []

    for line in lines[:20]:
        if ":" in line:
            question, answer = line.split(":", 1)
            cards.append({
                "question": question.strip() + "?",
                "answer": answer.strip(),
            })
        else:
            cards.append({
                "question": "این نکته را توضیح بده:",
                "answer": line,
            })

    return cards


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        '<div class="nova-brand">NOVA</div>',
        unsafe_allow_html=True,
    )
    st.caption("YOUR PERSONAL AI WORKSPACE")
    st.divider()

    page = st.radio(
        "فضای کاری",
        [
            "Chat",
            "Study",
            "Documents",
            "Web Research",
            "Export & Storage",
            "Settings",
        ],
        key="navigation",
    )

    st.divider()

    st.markdown("### وضعیت سیستم")

    api_available = bool(get_api_key()) and Groq is not None

    st.write(
        "مدل آنلاین: "
        + ("متصل به تنظیمات API" if api_available else "تنظیم نشده")
    )
    st.write(
        "جست‌وجوی وب: "
        + ("آماده" if DDGS is not None else "نیازمند نصب")
    )
    st.write(
        f"اسناد بارگذاری‌شده: "
        f"{len(st.session_state.documents)}"
    )
    st.write(
        f"پیام‌ها: {len(st.session_state.chat_history)}"
    )

    if st.button("پاک‌کردن تاریخچه گفتگو", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_answer = ""
        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="nova-brand">NOVA 2.0</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="nova-subtitle">'
    'AI Assistant · Study · Web · Documents · Export'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# CHAT PAGE
# =========================================================

if page == "Chat":
    st.subheader("گفت‌وگو با NOVA")

    mode = st.selectbox(
        "حالت پاسخ‌گویی",
        [
            "General Assistant",
            "Study Tutor",
            "Research",
            "Writing",
        ],
        index=[
            "General Assistant",
            "Study Tutor",
            "Research",
            "Writing",
        ].index(st.session_state.active_mode),
    )
    st.session_state.active_mode = mode

    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(
            role if role in ("user", "assistant") else "assistant"
        ):
            st.markdown(content)

    prompt = st.chat_input("پیام، سوال یا تمرین خود را برای NOVA بنویس...")

    if prompt:
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat(),
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("NOVA در حال تحلیل و آماده‌سازی پاسخ است..."):
                    # بررسی خودکار وب در صورتی که پاسخ در PDF نباشد یا سوال به صورت تمرین مطرح شود
                    web_results = []
                    greetings = ["سلام", "درود", "hello", "hi"]
                    is_greeting = any(g in prompt.lower() for g in greetings) and len(prompt.split()) <= 4
                    
                    if not is_greeting:
                        # جستجوی خودکار وب جهت پوشش تمرین‌ها و مباحث خارج از PDF
                        web_results = search_web(prompt, limit=3)

                    answer = generate_answer(prompt, mode, web_results=web_results)

                st.markdown(answer)

            except Exception as exc:
                answer = (
                    "در تولید پاسخ خطایی رخ داد. "
                    "تنظیمات API و اتصال شبکه را بررسی کنید.\n\n"
                    f"جزئیات: `{type(exc).__name__}: {exc}`"
                )
                st.error(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
        })

        st.session_state.last_answer = answer
        st.rerun()


# =========================================================
# DOCUMENTS PAGE
# =========================================================

elif page == "Documents":
    st.subheader("مدیریت اسناد PDF")

    uploaded_files = st.file_uploader(
        "فایل‌های PDF را انتخاب کن",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                data = uploaded_file.getvalue()
                signature = make_signature(data)
                filename = uploaded_file.name

                if (
                    st.session_state.document_hashes.get(filename)
                    == signature
                ):
                    continue

                with st.spinner(f"در حال پردازش {filename}..."):
                    pages = extract_pdf_text(data)

                st.session_state.documents[filename] = pages
                st.session_state.document_hashes[filename] = signature

                if not pages:
                    st.warning(
                        f"از {filename} متنی استخراج نشد. "
                        "ممکن است PDF اسکن‌شده باشد."
                    )
                else:
                    st.success(
                        f"{filename}: {len(pages)} صفحه متنی استخراج شد."
                    )

            except Exception as exc:
                st.error(f"خطا در {uploaded_file.name}: {exc}")

    st.divider()
    st.markdown("### اسناد فعال")

    if not st.session_state.documents:
        st.info("هنوز سندی بارگذاری نشده است.")
    else:
        for filename, pages in list(
            st.session_state.documents.items()
        ):
            with st.expander(f"{filename} · {len(pages)} صفحه"):
                for page in pages[:3]:
                    st.caption(f"صفحه {page['page']}")
                    st.write(page["text"][:1500])

                if len(pages) > 3:
                    st.caption("نمایش محدود به سه صفحه اول است.")

                if st.button(
                    f"حذف {filename}",
                    key=f"delete_{filename}",
                ):
                    del st.session_state.documents[filename]
                    st.session_state.document_hashes.pop(
                        filename, None
                    )
                    st.rerun()

    st.divider()
    st.markdown("### پرسش از اسناد")

    document_question = st.text_input(
        "سؤال درباره محتوای PDF یا تمرین‌ها",
        key="document_question",
    )

    if st.button("پاسخ از روی اسناد و وب", key="ask_document"):
        if not st.session_state.documents:
            st.warning("ابتدا یک PDF بارگذاری کن.")
        elif not document_question.strip():
            st.warning("سؤال را وارد کن.")
        else:
            try:
                with st.spinner("در حال تحلیل اسناد و وب..."):
                    web_results = search_web(document_question, limit=3)
                    answer = generate_answer(
                        document_question,
                        "Research",
                        web_results=web_results,
                    )

                st.markdown(answer)

                st.session_state.last_answer = answer
                st.session_state.chat_history.extend([
                    {
                        "role": "user",
                        "content": document_question,
                        "timestamp": datetime.now().isoformat(),
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().isoformat(),
                    },
                ])

            except Exception as exc:
                st.error(f"خطا: {exc}")


# =========================================================
# WEB RESEARCH PAGE
# =========================================================

elif page == "Web Research":
    st.subheader("جست‌وجوی وب")

    query = st.text_input(
        "موضوع موردنظر",
        placeholder="مثلاً: حل تمرین یا مقاله پژوهشی...",
    )

    limit = st.slider(
        "تعداد نتایج",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button("جست‌وجو", type="primary"):
        if not query.strip():
            st.warning("عبارت جست‌وجو را وارد کن.")
        else:
            try:
                with st.spinner("در حال جست‌وجوی وب..."):
                    results = search_web(query, limit)

                st.session_state.web_results = results

                if not results:
                    st.info("نتیجه‌ای پیدا نشد.")
                else:
                    for index, item in enumerate(results, start=1):
                        st.markdown(f"### {index}. {item['title']}")
                        st.write(item["snippet"])

                        if item["url"]:
                            st.markdown(
                                f"[مشاهده منبع]({item['url']})"
                            )

            except Exception as exc:
                st.error(
                    "جست‌وجو انجام نشد. اتصال شبکه و نصب "
                    f"کتابخانه را بررسی کن. جزئیات: {exc}"
                )

    if st.session_state.web_results:
        st.divider()

        if st.button("تحلیل نتایج با هوش مصنوعی"):
            try:
                with st.spinner("در حال تحلیل نتایج..."):
                    answer = generate_answer(
                        "نتایج جست‌وجوی موجود را تحلیل کن، "
                        "نکات مهم را خلاصه کن و پاسخ دقیق را ارائه بده.",
                        "Research",
                        st.session_state.web_results,
                    )

                st.markdown(answer)
                st.session_state.last_answer = answer

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as exc:
                st.error(f"خطا در تحلیل: {exc}")


# =========================================================
# STUDY PAGE
# =========================================================

elif page == "Study":
    st.subheader("فضای مطالعه")

    study_tab, flashcard_tab = st.tabs([
        "ساخت تمرین",
        "فلش‌کارت",
    ])

    with study_tab:
        study_text = st.text_area(
            "موضوع یا متن آموزشی یا صورت تمرین",
            height=220,
            placeholder="متن جزوه، سوال یا تمرین را وارد کن...",
        )

        study_task = st.selectbox(
            "نوع تمرین",
            [
                "خلاصه آموزشی",
                "حل تشریحی تمرین",
                "ساخت آزمون",
                "توضیح مرحله‌به‌مرحله",
            ],
        )

        if st.button("تولید محتوای آموزشی", type="primary"):
            if not study_text.strip():
                st.warning("ابتدا متن یا موضوع را وارد کن.")
            else:
                prompts = {
                    "خلاصه آموزشی":
                        "یک خلاصه دقیق و ساختاریافته تهیه کن.",
                    "حل تشریحی تمرین":
                        "این تمرین یا مسئله را به صورت کاملاً تشریحی، گام‌به‌گام و علمی حل کن.",
                    "ساخت آزمون":
                        "۱۰ سؤال چهارگزینه‌ای با پاسخ صحیح "
                        "و توضیح کوتاه تولید کن.",
                    "توضیح مرحله‌به‌مرحله":
                        "مطلب را از پایه و گام‌به‌گام آموزش بده.",
                }

                try:
                    with st.spinner("در حال آماده‌سازی پاسخ..."):
                        web_results = search_web(study_text, limit=3)
                        answer = generate_answer(
                            prompts[study_task] + "\n\n" + study_text,
                            "Study Tutor",
                            web_results=web_results,
                        )

                    st.markdown(answer)
                    st.session_state.last_answer = answer
                    st.session_state.quiz = [{
                        "prompt": study_task,
                        "content": answer,
                    }]

                except Exception as exc:
                    st.error(f"خطا: {exc}")

        if st.session_state.quiz:
            st.download_button(
                "دریافت محتوای آموزشی به‌صورت TXT",
                data="\n\n".join(
                    item["content"]
                    for item in st.session_state.quiz
                ),
                file_name="nova_study.txt",
                mime="text/plain",
            )

    with flashcard_tab:
        flashcard_text = st.text_area(
            "متن برای ساخت فلش‌کارت",
            height=160,
            key="flashcard_input",
        )

        if st.button("ساخت فلش‌کارت"):
            st.session_state.flashcards = build_flashcards(
                flashcard_text
            )

        for index, card in enumerate(
            st.session_state.flashcards
        ):
            with st.expander(
                f"کارت {index + 1}: {card['question']}"
            ):
                st.write(card["answer"])

        if st.session_state.flashcards:
            st.download_button(
                "خروجی فلش‌کارت‌ها (JSON)",
                data=json.dumps(
                    st.session_state.flashcards,
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="nova_flashcards.json",
                mime="application/json",
            )


# =========================================================
# EXPORT AND STORAGE PAGE
# =========================================================

elif page == "Export & Storage":
    st.subheader("خروجی و پشتیبان‌گیری")

    st.metric(
        "تعداد پیام‌ها",
        len(st.session_state.chat_history),
    )

    st.metric(
        "تعداد اسناد",
        len(st.session_state.documents),
    )

    st.download_button(
        "دانلود گفتگو به‌صورت Markdown",
        data=export_chat_markdown(),
        file_name=download_name("nova_chat", "md"),
        mime="text/markdown",
        use_container_width=True,
    )

    st.download_button(
        "دانلود گفتگو به‌صورت JSON",
        data=export_chat_json(),
        file_name=download_name("nova_chat", "json"),
        mime="application/json",
        use_container_width=True,
    )

    st.download_button(
        "دانلود گفتگو به‌صورت TXT",
        data="\n\n".join(
            f"{m['role'].upper()}:\n{m['content']}"
            for m in st.session_state.chat_history
        ),
        file_name=download_name("nova_chat", "txt"),
        mime="text/plain",
        use_container_width=True,
    )


# =========================================================
# SETTINGS PAGE
# =========================================================

elif page == "Settings":
    st.subheader("تنظیمات NOVA")

    model = st.selectbox(
        "مدل Groq",
        [
            "openai/gpt-oss-120b",
            "llama-3.1-70b-versatile",
            "llama3-8b-8192",
        ],
        index=0,
    )

    temperature = st.slider(
        "خلاقیت پاسخ",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.settings["temperature"]),
        step=0.1,
    )

    max_tokens = st.select_slider(
        "حداکثر طول پاسخ",
        options=[512, 1024, 2048, 4096],
        value=int(st.session_state.settings["max_tokens"]),
    )

    if st.button("ذخیره تنظیمات"):
        st.session_state.settings = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        st.success("تنظیمات این نشست ذخیره شد.")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "NOVA 2.0 · Local UI · Optional Online AI · "
    "Study · Documents · Web · Export"
)