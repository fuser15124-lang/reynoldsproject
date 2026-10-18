# NOVA 2 — دستیار هوشمند PDF، وب و مطالعه

نسخه ارتقایافته NOVA با رابط تیره/نئونی و رابط فارسی/انگلیسی. برنامه به‌صورت محلی روی رایانه اجرا می‌شود؛ قابلیت پاسخ‌گویی مدل زبانی و جست‌وجوی وب در صورت فعال‌کردن، از سرویس‌های آنلاین استفاده می‌کند.

## قابلیت‌ها

- جست‌وجو در چند PDF، بازیابی معنایی چندزبانه و استناد به شماره صفحه فیزیکی.
- حالت گفت‌وگو و تحلیل، معلم خصوصی، ساخت آزمون و فلش‌کارت.
- جست‌وجوی وب با Google Custom Search یا SerpAPI اختیاری و DuckDuckGo به‌عنوان مسیر جایگزین.
- تحلیل تصویر با مدل بینایی سازگار و کلید Groq.
- خروجی Markdown، متن ساده TXT و پشتیبان JSON.
- ابزارهای هسته‌ای مستقل از رابط کاربری و آزمون‌های واحد.
- کلیدها از Secrets یا متغیر محیطی خوانده می‌شوند؛ هیچ کلیدی در مخزن قرار ندهید.

## پیش‌نیازها

- Python 3.10 یا جدیدتر (پیشنهاد: 3.11)
- دسترسی اینترنت برای نصب وابستگی‌ها، دانلود مدل embedding و امکانات آنلاین
- برای پاسخ مدل زبانی: کلید API سرویس Groq

## نصب و اجرا

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

سپس آدرس محلی نشان‌داده‌شده در ترمینال (معمولاً `http://localhost:8501`) را باز کنید.

## تنظیم کلیدها

در مسیر `.streamlit/secrets.toml` فایل بسازید:

```toml
GROQ_API_KEY = "کلید-شما"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# اختیاری:
# GOOGLE_API_KEY = "..."
# GOOGLE_CSE_ID = "..."
# SERPAPI_API_KEY = "..."
```

یا متغیرهای محیطی هم‌نام را در محیط اجرا تعریف کنید. `.env.example` صرفاً نمونه است؛ برنامه به‌صورت خودکار فایل `.env` را بارگذاری نمی‌کند. فایل Secrets را هرگز commit نکنید. بدون کلید Groq، رابط و برخی امکانات محلی مانند آماده‌سازی/نمایه‌سازی PDF در دسترس‌اند، اما پاسخ‌گویی LLM و تحلیل تصویر کار نمی‌کنند.

## تست‌ها

تست‌های مستقل از شبکه و مدل:

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py core.py
```

برای تست دستی کامل: برنامه را اجرا کنید، یک PDF متنی بارگذاری و پردازش کنید، درخواست «صفحه ۲ را خلاصه کن» بدهید، سپس جست‌وجوی وب، حالت فلش‌کارت و سه نوع خروجی را امتحان کنید.

## محدودیت‌ها و دقت

- PDF اسکن‌شده یا تصویری ممکن است به OCR نیاز داشته باشد؛ نسخه فعلی متن قابل‌استخراج را پردازش می‌کند.
- بازیابی معنایی و نتایج وب ممکن است ناقص باشند. ادعاهای مهم را با منبع اصلی بررسی کنید.
- دسترسی و نام مدل‌ها به تنظیمات حساب ارائه‌دهنده بستگی دارد.
- اجرای رابط محلی است، اما درخواست‌های LLM، جست‌وجوی وب و تحلیل تصویر می‌توانند داده را به سرویس‌های خارجی بفرستند. برای اطلاعات حساس، سیاست‌های نگهداری داده ارائه‌دهنده را بررسی کنید و جست‌وجوی آنلاین را خاموش کنید.
- تاریخچه فعلی در session برنامه نگه‌داری می‌شود؛ برای نگه‌داری دائمی، از خروجی JSON استفاده کنید.

## ساختار پروژه

```text
app.py                 رابط Streamlit و اتصال سرویس‌ها
core.py                توابع خالص و قابل‌آزمون
requirements.txt        وابستگی‌ها
tests/test_core.py      تست‌های واحد
.streamlit/config.toml  تم تیره و تنظیمات محلی
.env.example            نمونه نام متغیرهای محیطی
```

## English summary

NOVA 2 is a locally hosted Streamlit assistant for multilingual PDF retrieval, web search, tutoring, quizzes, flashcards, image analysis, and Markdown/TXT/JSON conversation exports. Online model features require a configured Groq API key. See the Persian installation steps above; commands are the same for English users.
