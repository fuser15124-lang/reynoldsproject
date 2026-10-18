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
    page_title="NOVA AI Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
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
        "thinking": "NOVA در حال تحلیل و آماده‌سازی پاسخ است...",
        "error_msg": "در تولید پاسخ خطایی رخ داد. تنظیمات API و اتصال شبکه را بررسی کنید.",
        "upload_label": "فایل‌های PDF را انتخاب کن",
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
        "upload_label": "Choose PDF files",
        "ask_doc_input": "Question about PDF content or exercises",
        "ask_doc_btn": "Answer from Docs & Web",
        "study_text": "Subject, educational text, or exercise prompt",
        "generate_study": "Generate Study Content",
        "flashcard_text": "Text for creating flashcards",
        "make_flashcard": "Create Flashcards",
    },
    "🇪🇸 Español": {
        "title": "NOVA 2.0 · Asistente Inteligente",
        "subtitle": "IA Personal · Estudio · Web · Documentos · Exportar",
        "sidebar_workspace": "Espacio de trabajo",
        "sidebar_lang": "🌐 Seleccionar Idioma",
        "system_status": "Estado del Sistema",
        "online_model": "Modelo Online",
        "connected": "Conectado a la API",
        "not_configured": "No configurado",
        "web_search": "Búsqueda Web",
        "ready": "Listo",
        "need_install": "Requiere instalación",
        "uploaded_docs": "Documentos subidos",
        "messages_count": "Mensajes",
        "clear_chat": "Borrar historial de chat",
        "chat_page": "Chat",
        "study_page": "Estudio y Práctica",
        "docs_page": "Documentos",
        "web_page": "Investigación Web",
        "export_page": "Exportar y Copia de seguridad",
        "settings_page": "Configuración",
        "chat_input": "Escribe tu mensaje, pregunta o ejercicio para NOVA...",
        "thinking": "NOVA está analizando y preparando la respuesta...",
        "error_msg": "Ocurrió un error. Comprueba la configuración de la API.",
        "upload_label": "Elige archivos PDF",
        "ask_doc_input": "Pregunta sobre el contenido del PDF o ejercicios",
        "ask_doc_btn": "Responder desde documentos y web",
        "study_text": "Tema, texto educativo o enunciado de ejercicio",
        "generate_study": "Generar contenido de estudio",
        "flashcard_text": "Texto para crear tarjetas de memoria",
        "make_flashcard": "Crear tarjetas",
    },
    "🇮🇹 Italiano": {
        "title": "NOVA 2.0 · Assistente Intelligente",
        "subtitle": "IA Personale · Studio · Web · Documenti · Esporta",
        "sidebar_workspace": "Spazio di lavoro",
        "sidebar_lang": "🌐 Seleziona Lingua",
        "system_status": "Stato del Sistema",
        "online_model": "Modello Online",
        "connected": "Connesso all'API",
        "not_configured": "Non configurato",
        "web_search": "Ricerca Web",
        "ready": "Pronto",
        "need_install": "Richiede installazione",
        "uploaded_docs": "Documenti caricati",
        "messages_count": "Messaggi",
        "clear_chat": "Cancella cronologia chat",
        "chat_page": "Chat",
        "study_page": "Studio e Pratica",
        "docs_page": "Documenti",
        "web_page": "Ricerca Web",
        "export_page": "Esporta e Backup",
        "settings_page": "Impostazioni",
        "chat_input": "Scrivi il tuo messaggio, domanda o esercizio per NOVA...",
        "thinking": "NOVA sta analizzando e preparando la risposta...",
        "error_msg": "Si è verificato un errore. Controlla le impostazioni dell'API.",
        "upload_label": "Scegli file PDF",
        "ask_doc_input": "Domanda sul contenuto del PDF o esercizi",
        "ask_doc_btn": "Rispondi da documenti e web",
        "study_text": "Argomento, testo educativo o traccia dell'esercizio",
        "generate_study": "Genera contenuto di studio",
        "flashcard_text": "Testo per creare flashcard",
        "make_flashcard": "Crea flashcard",
    },
    "🇷🇺 Русский": {
        "title": "NOVA 2.0 · Умный помощник",
        "subtitle": "Личный ИИ · Учеба · Веб · Документы · Экспорт",
        "sidebar_workspace": "Рабочая область",
        "sidebar_lang": "🌐 Выбрать язык",
        "system_status": "Статус системы",
        "online_model": "Онлайн-модель",
        "connected": "Подключено к API",
        "not_configured": "Не настроено",
        "web_search": "Поиск в вебе",
        "ready": "Готово",
        "need_install": "Требуется установка",
        "uploaded_docs": "Загруженные документы",
        "messages_count": "Сообщения",
        "clear_chat": "Очистить историю чата",
        "chat_page": "Чат",
        "study_page": "Учеба и практика",
        "docs_page": "Документы",
        "web_page": "Веб-поиск",
        "export_page": "Экспорт и бэкап",
        "settings_page": "Настройки",
        "chat_input": "Введите сообщение, вопрос или упражнение для NOVA...",
        "thinking": "NOVA анализирует и готовит ответ...",
        "error_msg": "Произошла ошибка. Проверьте настройки API.",
        "upload_label": "Выберите PDF файлы",
        "ask_doc_input": "Вопрос по содержимому PDF или упражнениям",
        "ask_doc_btn": "Ответить по документам и вебу",
        "study_text": "Тема, учебный текст или условие упражнения",
        "generate_study": "Создать учебный материал",
        "flashcard_text": "Текст для создания карточек",
        "make_flashcard": "Создать карточки",
    },
    "🇹🇷 Türkçe": {
        "title": "NOVA 2.0 · Akıllı Asistan",
        "subtitle": "Kişisel Yapay Zeka · Çalışma · Web · Belgeler · Dışa Aktar",
        "sidebar_workspace": "Çalışma Alanı",
        "sidebar_lang": "🌐 Dil Seçin",
        "system_status": "Sistem Durumu",
        "online_model": "Çevrimiçi Model",
        "connected": "API'ye Bağlı",
        "not_configured": "Yapılandırılmadı",
        "web_search": "Web Arama",
        "ready": "Hazır",
        "need_install": "Kurulum Gerekli",
        "uploaded_docs": "Yüklenen Belgeler",
        "messages_count": "Mesajlar",
        "clear_chat": "Sohbet Geçmişini Temizle",
        "chat_page": "Sohbet",
        "study_page": "Çalışma ve Pratik",
        "docs_page": "Belgeler",
        "web_page": "Web Araştırma",
        "export_page": "Dışa Aktar ve Yedekle",
        "settings_page": "Ayarlar",
        "chat_input": "NOVA için mesajınızı, sorunuzu veya egzersizinizi yazın...",
        "thinking": "NOVA analiz ediyor و yanıt hazırlıyor...",
        "error_msg": "Bir hata oluştu. API ayarlarını ve ağ bağlantısını kontrol edin.",
        "upload_label": "PDF dosyalarını seçin",
        "ask_doc_input": "PDF içeriği یا alıştırmalar hakkında soru",
        "ask_doc_btn": "Belgelerden ve Web'den Yanıtla",
        "study_text": "Konu, eğitim metni یا egzersiz metni",
        "generate_study": "Eğitim İçeriği Oluştur",
        "flashcard_text": "Bilgi kartı oluşturmak için metin",
        "make_flashcard": "Bilgi Kartı Oluştur",
    },
    "🇨🇳 中文": {
        "title": "NOVA 2.0 · 智能助手",
        "subtitle": "个人AI · 学习 · 网页 · 文档 · 导出",
        "sidebar_workspace": "工作区",
        "sidebar_lang": "🌐 选择语言",
        "system_status": "系统状态",
        "online_model": "在线模型",
        "connected": "已连接API",
        "not_configured": "未配置",
        "web_search": "网页搜索",
        "ready": "就绪",
        "need_install": "需要安装",
        "uploaded_docs": "已上传文档",
        "messages_count": "消息数",
        "clear_chat": "清除聊天记录",
        "chat_page": "聊天",
        "study_page": "学习与练习",
        "docs_page": "文档管理",
        "web_page": "网页研究",
        "export_page": "导出与备份",
        "settings_page": "设置",
        "chat_input": "向 NOVA 发送消息、提问或联系练习...",
        "thinking": "NOVA 正在分析并准备回答...",
        "error_msg": "发生错误，请检查API设置和网络连接。",
        "upload_label": "选择 PDF 文件",
        "ask_doc_input": "关于PDF内容或练习的问题",
        "ask_doc_btn": "从文档和网页回答",
        "study_text": "主题、教育文本或练习提示",
        "generate_study": "生成学习内容",
        "flashcard_text": "用于制作抽认卡的文本",
        "make_flashcard": "制作抽认卡",
    },
    "🇯🇵 日本語": {
        "title": "NOVA 2.0 · スマートアシスタント",
        "subtitle": "パーソナルAI · 学習 · ウェブ · 文書 · エクスポート",
        "sidebar_workspace": "ワークスペース",
        "sidebar_lang": "🌐 言語選択",
        "system_status": "システムステータス",
        "online_model": "オンラインモデル",
        "connected": "API接続済み",
        "not_configured": "未設定",
        "web_search": "ウェブ検索",
        "ready": "準備完了",
        "need_install": "インストールが必要",
        "uploaded_docs": "アップロードされた文書",
        "messages_count": "メッセージ数",
        "clear_chat": "チャット履歴をクリア",
        "chat_page": "チャット",
        "study_page": "学習と演習",
        "docs_page": "文書管理",
        "web_page": "ウェブ調査",
        "export_page": "エクスポートとバックアップ",
        "settings_page": "設定",
        "chat_input": "NOVAへのメッセージ、質問、または演習を入力してください...",
        "thinking": "NOVAが分析し、回答を準備しています...",
        "error_msg": "エラーが発生しました。API設定とネットワーク接続を確認してください。",
        "upload_label": "PDFファイルを選択",
        "ask_doc_input": "PDFの内容や演習に関する質問",
        "ask_doc_btn": "文書とウェブから回答",
        "study_text": "トピック、教育テキスト、または演習のプロンプト",
        "generate_study": "学習コンテンツを生成",
        "flashcard_text": "単語帳（フラッシュカード）作成用のテキスト",
        "make_flashcard": "フラッシュカードを作成",
    },
    "🇮🇱 עברית": {
        "title": "NOVA 2.0 · עוזר חכם",
        "subtitle": "בינה מלאכותית אישית · לימודים · אינטרנט · מסמכים · ייצוא",
        "sidebar_workspace": "מרחב עבודה",
        "sidebar_lang": "🌐 בחר שפה",
        "system_status": "מצב המערכת",
        "online_model": "מודל מקוון",
        "connected": "מחובר ל-API",
        "not_configured": "לא מוגדר",
        "web_search": "חיפוש באינטרנט",
        "ready": "מוכן",
        "need_install": "דורש התקנה",
        "uploaded_docs": "מסמכים שהועלו",
        "messages_count": "הודעות",
        "clear_chat": "נקה היסטוריית שיחות",
        "chat_page": "צ'אט",
        "study_page": "למידה ותרגול",
        "docs_page": "ניהול מסמכים",
        "web_page": "מחקר אינטרנט",
        "export_page": "ייצוא וגיבוי",
        "settings_page": "הגדרות",
        "chat_input": "הקלד הודעה, שאלה או תרגיל עבור NOVA...",
        "thinking": "NOVA מנתח ומכין את התשובה...",
        "error_msg": "אירעה שגיאה. בדוק את הגדרות ה-API ואת חיבור הרשת.",
        "upload_label": "בחר קבצי PDF",
        "ask_doc_input": "שאלה לגבי תוכן ה-PDF או תרגילים",
        "ask_doc_btn": "ענה מתוך המסמכים והאינטרנט",
        "study_text": "נושא, טקסט לימודי أو תיאור תרגיל",
        "generate_study": "צור תוכן לימודי",
        "flashcard_text": "טקסט ליצירת כרטיסיות לימוד",
        "make_flashcard": "צור כרטיסיות",
    }
}


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
    "language": "🇮🇷 فارسی",
    "settings": {
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
# STYLING (Supports RTL for Persian and Hebrew)
# =========================================================

current_lang = st.session_state.get("language", "🇮🇷 فارسی")
is_rtl = "فارسی" in current_lang or "عברית" in current_lang

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(ellipse at top left,
                rgba(37, 60, 105, .28), transparent 45%),
            #080b14;
        color: #e7edf7;
        direction: {"rtl" if is_rtl else "ltr"};
    }}

    [data-testid="stSidebar"] {{
        background: #0c1120;
        border-right: 1px solid #252e49;
        direction: {"rtl" if is_rtl else "ltr"};
    }}

    h1, h2, h3 {{
        color: #e8faff !important;
    }}

    .nova-brand {{
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: .16em;
        color: #5cecff;
        text-shadow: 0 0 22px rgba(92, 236, 255, .3);
    }}

    .nova-subtitle {{
        color: #9baac5;
        font-size: .95rem;
        margin-bottom: 1.5rem;
    }}

    div[data-testid="stChatMessage"] {{
        border: 1px solid #27334e;
        border-radius: 14px;
        background: rgba(15, 22, 39, .72);
        padding: 12px;
        text-align: {"right" if is_rtl else "left"};
    }}

    .stButton button {{
        border: 1px solid #354b6d;
        border-radius: 10px;
        transition: all .2s ease;
    }}

    .stButton button:hover {{
        border-color: #5cecff;
        color: #5cecff;
    }}

    div[data-testid="stMetric"] {{
        background: #10182a;
        padding: 14px;
        border: 1px solid #26334e;
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CONFIGURATION AND HELPERS
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
    remaining = 24000
    for filename, pages in st.session_state.documents.items():
        if remaining <= 0:
            break
        for page in pages:
            if remaining <= 0:
                break
            section = f"\n[Document: {filename} | Page: {page['page']}]\n{page['text']}\n"
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
You are NOVA, a friendly, warm, and helpful AI assistant.
Respond warmly, politely, and naturally in {selected_language}.
Offer your assistance with their PDF documents, exercises, questions, or general queries.
"""
    else:
        system_prompt = f"""
You are NOVA, an advanced expert AI academic tutor, researcher, and multilingual assistant.

Rules:
- Respond strictly in the following language chosen by the user: {selected_language}.
- Distinguish verified evidence from uncertainty.
- Never invent citations, URLs, quotations, or page numbers.
- **First Priority (PDF Context):** If document context is supplied, prioritize it for document/study questions. Cite exact evidence and page numbers using [Document: filename, Page: N] or [صفحه N].
- **Second Priority (Web Search Context / Exercises):** If a question is an exercise or its answer is missing from the document context, use the provided web search results to give a fully correct, well-structured, and accurate answer.
- For study tasks and exercises, explain concepts step by step in {selected_language}.
"""

        if mode == "Study Tutor":
            system_prompt += "\nAct as a study tutor. Explain reasoning, give examples, and provide step-by-step solutions to exercises."
        elif mode == "Research":
            system_prompt += "\nAct as a research assistant. Compare evidence and separate facts and interpretations."
        elif mode == "Writing":
            system_prompt += "\nAct as a writing assistant. Produce clear, structured text."

    context = get_document_context()
    if context and not is_greeting:
        context = "\n\nUPLOADED DOCUMENT EXCERPTS:\n" + context

    if not is_greeting and web_results:
        context += "\n\nWEB SEARCH RESULTS (for exercises / missing info):\n"
        for index, item in enumerate(web_results, start=1):
            context += f"\n[{index}] {item['title']}\nURL: {item['url']}\nExcerpt: {item['snippet']}\n"

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
        return f"Error communicating with AI: {e}"


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


def build_flashcards(text: str) -> list[dict]:
    lines = [line.strip(" -*\t") for line in text.splitlines() if len(line.strip()) >= 20]
    cards = []
    for line in lines[:20]:
        if ":" in line:
            question, answer = line.split(":", 1)
            cards.append({"question": question.strip() + "?", "answer": answer.strip()})
        else:
            cards.append({"question": "Explain this concept:", "answer": line})
    return cards


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="nova-brand">NOVA</div>', unsafe_allow_html=True)
    st.caption("YOUR PERSONAL AI WORKSPACE")
    st.divider()

    languages_list = [
        "🇮🇷 فارسی",
        "🇺🇸 English",
        "🇪🇸 Español",
        "🇮🇹 Italiano",
        "🇷🇺 Русский",
        "🇹🇷 Türkçe",
        "🇨🇳 中文",
        "🇯🇵 日本語",
        "🇮🇱 עברית",
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

st.markdown(f'<div class="nova-brand">{t("title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="nova-subtitle">{t("subtitle")}</div>', unsafe_allow_html=True)


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

    prompt = st.chat_input(t("chat_input"))

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