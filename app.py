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
    page_title="NOVA 3.0 AI Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# MULTILINGUAL DICTIONARIES
# =========================================================

TEXTS = {
    "🇮🇷 فارسی": {
        "title": "NOVA 3.0",
        "subtitle": "دستیار آموزشی پیشرفته و معلم خصوصی هوشمند",
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
        "chat_page": "گفت‌وگو و آموزش",
        "study_page": "مطالعه و حل تمرین",
        "docs_page": "مدیریت اسناد",
        "web_page": "جست‌وجوی وب",
        "export_page": "خروجی و پشتیبان",
        "settings_page": "تنظیمات",
        "chat_input": "پیام، سوال درسی، یا تمرین خود را برای NOVA بنویس...",
        "thinking": "NOVA در حال تحلیل دقیق سؤال و آماده‌سازی پاسخ است...",
        "error_msg": "در تولید پاسخ خطایی رخ داد. تنظیمات API و اتصال شبکه را بررسی کنید.",
        "upload_label": "افزودن فایل/عکس",
        "ask_doc_input": "سؤال درباره محتوای PDF یا تمرین‌ها",
        "ask_doc_btn": "پاسخ کامل و گام‌به‌گام از روی اسناد",
        "study_text": "موضوع یا متن آموزشی یا صورت تمرین",
        "generate_study": "تولید محتوای آموزشی",
    },
    "🇺🇸 English": {
        "title": "NOVA 3.0",
        "subtitle": "Advanced Educational Assistant & Smart Tutor",
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
        "chat_page": "Chat & Tutoring",
        "study_page": "Study & Practice",
        "docs_page": "Documents",
        "web_page": "Web Research",
        "export_page": "Export & Backup",
        "settings_page": "Settings",
        "chat_input": "Type your message, study question, or exercise...",
        "thinking": "NOVA is carefully analyzing the question and preparing the response...",
        "error_msg": "An error occurred. Check API settings and network connection.",
        "upload_label": "Add File/Image",
        "ask_doc_input": "Question about PDF content or exercises",
        "ask_doc_btn": "Answer from Documents",
        "study_text": "Subject, educational text, or exercise prompt",
        "generate_study": "Generate Study Content",
    },
    "🇪🇸 Español": {
        "title": "NOVA 3.0",
        "subtitle": "Asistente Educativo Avanzado y Tutor Inteligente",
        "sidebar_workspace": "Espacio de trabajo",
        "sidebar_lang": "🌐 Seleccionar Idioma",
        "new_chat": "Nuevo Chat",
        "system_status": "Estado del Sistema",
        "online_model": "Modelo Online",
        "connected": "Conectado a la API",
        "not_configured": "No configurado",
        "web_search": "Búsqueda Web",
        "ready": "Listo",
        "need_install": "Requiere instalación",
        "uploaded_docs": "Documentos subidos",
        "messages_count": "Mensajes",
        "clear_chat": "Borrar historial",
        "chat_page": "Chat y Tutoría",
        "study_page": "Estudio y Práctica",
        "docs_page": "Documentos",
        "web_page": "Investigación Web",
        "export_page": "Exportar",
        "settings_page": "Configuración",
        "chat_input": "Escribe tu mensaje o pregunta para NOVA...",
        "thinking": "NOVA está analizando la pregunta...",
        "error_msg": "Ocurrió un error.",
        "upload_label": "Añadir archivo",
        "ask_doc_input": "Pregunta sobre el PDF",
        "ask_doc_btn": "Responder",
        "study_text": "Tema o texto",
        "generate_study": "Generar contenido",
    },
    "🇮🇹 Italiano": {
        "title": "NOVA 3.0",
        "subtitle": "Assistente Educativo Avanzato e Tutore Intelligente",
        "sidebar_workspace": "Spazio di lavoro",
        "sidebar_lang": "🌐 Seleziona Lingua",
        "new_chat": "Nuova Chat",
        "system_status": "Stato del Sistema",
        "online_model": "Modello Online",
        "connected": "Connesso all'API",
        "not_configured": "Non configurato",
        "web_search": "Ricerca Web",
        "ready": "Pronto",
        "need_install": "Richiede installazione",
        "uploaded_docs": "Documenti caricati",
        "messages_count": "Messaggi",
        "clear_chat": "Cancella cronologia",
        "chat_page": "Chat e Tutorato",
        "study_page": "Studio e Pratica",
        "docs_page": "Documenti",
        "web_page": "Ricerca Web",
        "export_page": "Esporta",
        "settings_page": "Impostazioni",
        "chat_input": "Scrivi il tuo messaggio per NOVA...",
        "thinking": "NOVA sta analizzando la domanda...",
        "error_msg": "Si è verificato un errore.",
        "upload_label": "Aggiungi file",
        "ask_doc_input": "Domanda sul PDF",
        "ask_doc_btn": "Rispondi",
        "study_text": "Argomento o testo",
        "generate_study": "Genera contenuto",
    },
    "🇷🇺 Русский": {
        "title": "NOVA 3.0",
        "subtitle": "Продвинутый образовательный помощник",
        "sidebar_workspace": "Рабочая область",
        "sidebar_lang": "🌐 Выбрать язык",
        "new_chat": "Новый чат",
        "system_status": "Статус системы",
        "online_model": "Онлайн-модель",
        "connected": "Подключено к API",
        "not_configured": "Не настроено",
        "web_search": "Поиск в вебе",
        "ready": "Готово",
        "need_install": "Требуется установка",
        "uploaded_docs": "Загруженные документы",
        "messages_count": "Сообщения",
        "clear_chat": "Очистить историю",
        "chat_page": "Чат и обучение",
        "study_page": "Учеба и практика",
        "docs_page": "Документы",
        "web_page": "Веб-поиск",
        "export_page": "Экспорт",
        "settings_page": "Настройки",
        "chat_input": "Введите сообщение для NOVA...",
        "thinking": "NOVA анализирует вопрос...",
        "error_msg": "Произошла ошибка.",
        "upload_label": "Добавить файл",
        "ask_doc_input": "Вопрос по PDF",
        "ask_doc_btn": "Ответить",
        "study_text": "Тема или текст",
        "generate_study": "Создать материал",
    },
    "🇹🇷 Türkçe": {
        "title": "NOVA 3.0",
        "subtitle": "Gelişmiş Eğitim Asistanı ve Akıllı Eğitmen",
        "sidebar_workspace": "Çalışma Alanı",
        "sidebar_lang": "🌐 Dil Seçin",
        "new_chat": "Yeni Sohbet",
        "system_status": "Sistem Durumu",
        "online_model": "Çevrimiçi Model",
        "connected": "API'ye Bağlı",
        "not_configured": "Yapılandırılmadı",
        "web_search": "Web Arama",
        "ready": "Hazır",
        "need_install": "Kurulum Gerekli",
        "uploaded_docs": "Yüklenen Belgeler",
        "messages_count": "Mesajlar",
        "clear_chat": "Geçmişi Temizle",
        "chat_page": "Sohbet",
        "study_page": "Çalışma",
        "docs_page": "Belgeler",
        "web_page": "Web Araştırma",
        "export_page": "Dışa Aktar",
        "settings_page": "Ayarlar",
        "chat_input": "NOVA için mesajınızı yazın...",
        "thinking": "NOVA analiz ediyor...",
        "error_msg": "Bir hata oluştu.",
        "upload_label": "Dosya Ekle",
        "ask_doc_input": "PDF hakkında soru",
        "ask_doc_btn": "Yanıtla",
        "study_text": "Konu veya metin",
        "generate_study": "İçerik Oluştur",
    },
    "🇨🇳 中文": {
        "title": "NOVA 3.0",
        "subtitle": "高级教育助手与智能导师",
        "sidebar_workspace": "工作区",
        "sidebar_lang": "🌐 选择语言",
        "new_chat": "新对话",
        "system_status": "系统状态",
        "online_model": "在线模型",
        "connected": "已连接API",
        "not_configured": "未配置",
        "web_search": "网页搜索",
        "ready": "就绪",
        "need_install": "需要安装",
        "uploaded_docs": "已上传文档",
        "messages_count": "消息数",
        "clear_chat": "清除历史",
        "chat_page": "聊天与辅导",
        "study_page": "学习与练习",
        "docs_page": "文档",
        "web_page": "网页研究",
        "export_page": "导出",
        "settings_page": "设置",
        "chat_input": "向 NOVA 发送消息...",
        "thinking": "NOVA 正在分析问题...",
        "error_msg": "发生错误。",
        "upload_label": "添加文件",
        "ask_doc_input": "关于PDF的问题",
        "ask_doc_btn": "回答",
        "study_text": "主题或文本",
        "generate_study": "生成内容",
    },
    "🇯🇵 日本語": {
        "title": "NOVA 3.0",
        "subtitle": "高度な教育アシスタント・スマートチューター",
        "sidebar_workspace": "ワークスペース",
        "sidebar_lang": "🌐 言語選択",
        "new_chat": "新しいチャット",
        "system_status": "システム状態",
        "online_model": "オンラインモデル",
        "connected": "API接続済み",
        "not_configured": "未設定",
        "web_search": "ウェブ検索",
        "ready": "準備完了",
        "need_install": "インストールが必要",
        "uploaded_docs": "アップロードされた文書",
        "messages_count": "メッセージ数",
        "clear_chat": "履歴をクリア",
        "chat_page": "チャット",
        "study_page": "学習と演習",
        "docs_page": "文書管理",
        "web_page": "ウェブ調査",
        "export_page": "エクスポート",
        "settings_page": "設定",
        "chat_input": "NOVAへのメッセージを入力...",
        "thinking": "NOVAが分析中...",
        "error_msg": "エラーが発生しました。",
        "upload_label": "ファイル追加",
        "ask_doc_input": "PDFに関する質問",
        "ask_doc_btn": "回答",
        "study_text": "トピックまたはテキスト",
        "generate_study": "コンテンツ生成",
    },
    "🇮🇱 עברית": {
        "title": "NOVA 3.0",
        "subtitle": "עוזר חינוכי מתקדם ומורה פרטי חכם",
        "sidebar_workspace": "מרחב עבודה",
        "sidebar_lang": "🌐 בחר שפה",
        "new_chat": "שיחה חדשה",
        "system_status": "מצב מערכת",
        "online_model": "מודל מקוון",
        "connected": "מחובר ל-API",
        "not_configured": "לא מוגדר",
        "web_search": "חיפוש אינטרנט",
        "ready": "מוכן",
        "need_install": "דורש התקנה",
        "uploaded_docs": "מסמכים שהועלו",
        "messages_count": "הודעות",
        "clear_chat": "נקה היסטוריה",
        "chat_page": "צ'אט ולמידה",
        "study_page": "למידה ותרגול",
        "docs_page": "מסמכים",
        "web_page": "מחקר אינטרנט",
        "export_page": "ייצוא",
        "settings_page": "הגדרות",
        "chat_input": "הקלד הודעה עבור NOVA...",
        "thinking": "NOVA מנתח את השאלה...",
        "error_msg": "אירעה שגיאה.",
        "upload_label": "הוסף קובץ",
        "ask_doc_input": "שאלה على ال-PDF",
        "ask_doc_btn": "ענה",
        "study_text": "נושא أو טקסט",
        "generate_study": "צור תוכן",
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
    "language": "🇮🇷 فارسی",
    "settings": {
        "model": "llama3-70b-8192",  # مدل بسیار پایدار و سراسری Groq
        "temperature": 0.2,
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
        font-size: 1.6rem;
        font-weight: 800;
        color: #5cecff;
        margin-bottom: 0.1rem;
    }}

    .nova-header-sub {{
        color: #9baac5;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }}

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

    code, pre {{
        background-color: transparent !important;
        border: none !important;
        color: #e7edf7 !important;
        font-family: Tahoma, sans-serif !important;
        font-size: 1rem !important;
        padding: 0 !important;
        white-space: pre-wrap !important;
    }}

    .stButton button {{
        border: 1px solid #354b6d;
        border-radius: 8px;
        width: 100%;
    }}

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
    remaining = 5000  # کنترل بهینه توکن‌ها
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


def search_web(query: str, limit: int = 3) -> list[dict]:
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
تو هوش مصنوعی آموزشی پیشرفته نسخه ۳.۰ به نام NOVA هستی. به گرمی و با ادب به زبان {selected_language} جواب بده.
"""
    else:
        system_prompt = f"""
SYSTEM PROMPT — هوش مصنوعی آموزشی حرفه‌ای (NOVA 3.0)
تو یک معلم خصوصی و دستیار هوشمند هستی.
۱. تمامی فرمول‌ها و روابط را به صورت روان، ساده و با علائم ریاضی استاندارد (× و ÷) بنویس. مطلقاً از کادرهای کدنویسی استفاده نکن.
۲. تمام سوالات را به زبان {selected_language} گام‌به‌گام و کامل حل کن.
۳. اولویت اول با متن اسناد بارگذاری‌شده است؛ اگر فایلی هست، پاسخ را دقیقاً از روی آن استخراج کن.
"""

    context = get_document_context()
    if context and not is_greeting:
        context = "\n\nمتن اسناد بارگذاری‌شده:\n" + context

    if not is_greeting and web_results:
        context += "\n\nنتایج جستجوی وب:\n"
        for index, item in enumerate(web_results, start=1):
            context += f"\n[{index}] عنوان: {item['title']}\nمتن: {item['snippet']}\n"

    messages = [{"role": "system", "content": system_prompt}]
    if context and not is_greeting:
        messages.append({"role": "system", "content": context})

    for message in st.session_state.chat_history[-4:]:
        if message.get("role") in ("user", "assistant"):
            messages.append({"role": message["role"], "content": message.get("content", "")})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=settings["model"],
            messages=messages,
            temperature=float(settings["temperature"]),
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"خطا در ارتباط با هوش مصنوعی: {e}"


def download_name(prefix: str, extension: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def export_chat_markdown() -> str:
    lines = ["# NOVA 3.0 Conversation", "", f"Exported: {datetime.now().isoformat(timespec='seconds')}", ""]
    for message in st.session_state.chat_history:
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        lines.extend([f"## {role}", "", content, ""])
    return "\n".join(lines)


def export_chat_json() -> str:
    payload = {
        "application": "NOVA",
        "version": "3.0",
        "exported_at": datetime.now().isoformat(),
        "messages": st.session_state.chat_history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("### NOVA 3.0")
    st.divider()

    if st.button(t("new_chat"), use_container_width=True, type="primary"):
        st.session_state.chat_history = []
        st.session_state.last_answer = ""
        st.rerun()

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
    model = st.selectbox("Model", ["llama3-70b-8192", "mixtral-8x7b-32768"], index=0)
    if st.button("Save Settings"):
        st.session_state.settings["model"] = model
        st.success("Settings saved.")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("NOVA 3.0 · Professional Educational AI Assistant")