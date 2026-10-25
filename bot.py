import os
import io
import re
import tempfile
import threading
import requests
from dotenv import load_dotenv
import telebot
from telebot import apihelper
from telebot.types import Message
import sys
import fitz

# Load settings before importing modules that read config.py.
load_dotenv()

# Import the existing translation logic
from pdf_processor import extract_content, generate_documents, process_as_slides, is_scanned_pdf, integrate_translations, generate_overlay_pdf, generate_bilingual_overlay_pdf
from translator import GeminiQuotaExceeded, extract_and_translate_page, generate_study_material

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
BOT_OWNER_ID = os.getenv("BOT_OWNER_ID", "").strip()

if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "PUT_YOUR_TOKEN_HERE":
    print("Error: TELEGRAM_BOT_TOKEN is not set in .env")
    sys.exit(1)

TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "").strip()
if TELEGRAM_PROXY:
    apihelper.proxy = {'http': TELEGRAM_PROXY, 'https': TELEGRAM_PROXY}
    # Never print a proxy URL: it may contain credentials.
    print("Telegram proxy is configured.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


def _new_temp_file_path(original_name):
    """Create a unique temporary path using only the uploaded file extension."""
    suffix = os.path.splitext(os.path.basename(original_name))[1][:12] or ".tmp"
    fd, path = tempfile.mkstemp(prefix="pdf-translator-", suffix=suffix)
    os.close(fd)
    return path


def _cleanup_temp_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


class TelegramFileDownloadError(Exception):
    """Safe, user-facing error for Telegram file retrieval failures."""


def _download_telegram_file(file_id, status=None, expected_size=None):
    """Download a Telegram file with finite network timeouts and size checks.

    pyTelegramBotAPI's download_file() does not pass a timeout to requests.get,
    so a stalled connection can leave the processing thread stuck forever.
    """
    if status:
        status.info("جارٍ الاتصال بخوادم Telegram لاسترجاع الملف...")

    try:
        file_info = bot.get_file(file_id)
    except Exception as exc:
        raise TelegramFileDownloadError(
            f"تعذر استرجاع بيانات الملف من Telegram ({type(exc).__name__}). أعد المحاولة."
        ) from None

    file_path = getattr(file_info, "file_path", None)
    if not file_path:
        raise TelegramFileDownloadError("لم يُرجع Telegram مسارًا صالحًا للملف.")

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if expected_size and expected_size > max_bytes:
        raise TelegramFileDownloadError(
            f"حجم الملف يتجاوز الحد المسموح ({MAX_FILE_SIZE_MB}MB)."
        )
    download_timeout = 35
    max_attempts = 3
    if status:
        size_label = f" ({expected_size / 1024:.0f}KB)" if expected_size else ""
        status.info(
            f"جارٍ تنزيل الملف{size_label} من Telegram؛ ستتم إعادة المحاولة تلقائيًا عند الانقطاع..."
        )

    url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        deadline_expired = threading.Event()
        downloaded = 0
        try:
            with requests.get(
                url,
                stream=True,
                timeout=(10, 15),
                proxies=apihelper.proxy or None,
            ) as response:
                response.raise_for_status()
                chunks = []

                def _abort_slow_download():
                    deadline_expired.set()
                    response.close()

                deadline_timer = threading.Timer(download_timeout, _abort_slow_download)
                deadline_timer.daemon = True
                deadline_timer.start()
                try:
                    for chunk in response.iter_content(chunk_size=16 * 1024):
                        if deadline_expired.is_set():
                            raise TimeoutError("Telegram file download exceeded its deadline")
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise TelegramFileDownloadError(
                                f"حجم الملف يتجاوز الحد المسموح ({MAX_FILE_SIZE_MB}MB)."
                            )
                        chunks.append(chunk)
                    if deadline_expired.is_set():
                        raise TimeoutError("Telegram file download exceeded its deadline")
                finally:
                    deadline_timer.cancel()

            data = b"".join(chunks)
            if not data:
                raise TelegramFileDownloadError("وصل ملف فارغ من Telegram؛ أعد إرسال الملف.")
            if expected_size and downloaded != expected_size:
                raise requests.ConnectionError(
                    f"incomplete download: received {downloaded} of {expected_size} bytes"
                )
            return data
        except TelegramFileDownloadError:
            raise
        except (requests.RequestException, TimeoutError) as exc:
            last_error = (
                TimeoutError("overall download deadline expired")
                if deadline_expired.is_set() else exc
            )
            # Requests exceptions can include the full Telegram file URL, which
            # contains the bot token. Keep logs useful without leaking secrets.
            detail = str(exc)
            if TELEGRAM_BOT_TOKEN:
                detail = detail.replace(TELEGRAM_BOT_TOKEN, "<redacted>")
            if TELEGRAM_PROXY:
                detail = detail.replace(TELEGRAM_PROXY, "<proxy>")
            detail = " ".join(detail.split())[:180]
            print(
                f"[TELEGRAM DOWNLOAD] attempt {attempt}/{max_attempts} failed: "
                f"{type(exc).__name__}; received {locals().get('downloaded', 0)} bytes; "
                f"{detail}"
            )
            if attempt < max_attempts:
                if status:
                    status.info(
                        f"تعذر تنزيل الملف مؤقتًا؛ إعادة المحاولة {attempt + 1}/{max_attempts}..."
                    )
                threading.Event().wait(attempt)

    error_text = str(last_error).lower() if last_error else ""
    if "proxy" in error_text or "tunnel connection" in error_text:
        error_kind = "ProxyError"
    elif "timed out" in error_text or "deadline" in error_text:
        error_kind = "Timeout"
    elif "name or service not known" in error_text or "getaddrinfo" in error_text:
        error_kind = "DNS/network error"
    elif "connection reset" in error_text or "remote end closed" in error_text:
        error_kind = "ConnectionDropped"
    else:
        error_kind = type(last_error).__name__ if last_error else "UnknownError"
    raise TelegramFileDownloadError(
        f"فشل تنزيل الملف من Telegram بعد {max_attempts} محاولات ({error_kind}). "
        "تحقق من اتصال الجهاز أو إعداد TELEGRAM_PROXY ثم أعد المحاولة."
    ) from None

# Mocked classes to replace Streamlit's UI components
class TelegramProgressMock:
    def __init__(self, bot, chat_id, message_id):
        self._bot = bot
        self._chat_id = chat_id
        self._message_id = message_id

    def progress(self, value):
        # We can safely ignore continuous progress bar updates in Telegram to avoid rate limits
        pass

class TelegramStatusMock:
    def __init__(self, bot, chat_id, text="جاري المعالجة...", message_id=None):
        self._bot = bot
        self._chat_id = chat_id
        self._last_text = ""
        if message_id:
            self.message_id = message_id
            self._update_text(text)
        else:
            self._msg = bot.send_message(chat_id, text)
            self.message_id = self._msg.message_id
            self._last_text = text

    def _update_text(self, text):
        if text != self._last_text:
            try:
                self._bot.edit_message_text(text, chat_id=self._chat_id, message_id=self.message_id)
                self._last_text = text
            except Exception as exc:
                # Keep the user informed even if Telegram refuses to edit the
                # existing status message (for example, after a network error).
                print(f"[STATUS] edit failed ({type(exc).__name__}); sending a new status message")
                try:
                    msg = self._bot.send_message(self._chat_id, text)
                    self.message_id = msg.message_id
                    self._last_text = text
                except Exception as send_exc:
                    print(f"[STATUS] send failed ({type(send_exc).__name__})")

    def info(self, text):
         self._update_text("ℹ️ " + text)

    def success(self, text):
         self._update_text("✅ " + text)

    def error(self, text):
         self._update_text("❌ " + text)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "مرحباً بك في بوت الترجمة! 🎓\n\n"
        "📄 أرسل لي ملف **PDF** أو **PPTX** وسأقوم بترجمته إلى العربية.\n"
        "📸 أو أرسل **صورة مباشرة** (صفحة كتاب مثلاً) وسأترجمها فوراً!\n\n"
        "✨ يتم كشف نوع الملف (نصي/مصوّر) **تلقائياً**.\n\n"
        "ملاحظة: المعالجة قد تستغرق بعض الوقت حسب حجم الملف."
    )
    bot.reply_to(message, welcome_text, parse_mode='Markdown')

# Cache for storing user files temporarily between upload and option selection
# Format: chat_id -> (document, timestamp)
user_files_cache = {}
_CACHE_TTL_SEC = 300  # 5 minutes

# Extracted lecture text remains available briefly after translation so users
# can request a quiz or mind map without uploading the file again.
_study_material_cache = {}
_STUDY_CACHE_TTL_SEC = 3600
_active_study_jobs = set()
_active_study_jobs_lock = threading.Lock()


def _study_source_text(content_items):
    parts = []
    for item in content_items:
        if item.get("type") == "text":
            text = item.get("plain", "")
            if text and text.strip():
                parts.append(text.strip())
        elif item.get("type") == "table":
            rows = item.get("rows", [])
            for row in rows:
                cells = row.get("cells", row) if isinstance(row, dict) else row
                if isinstance(cells, (list, tuple)):
                    values = [str(cell).strip() for cell in cells if str(cell).strip()]
                    if values:
                        parts.append(" | ".join(values))
    return "\n\n".join(parts)[:60000]


def _study_page_texts(content_items):
    pages = {}
    for item in content_items:
        page_num = item.get("page_num")
        if page_num is None:
            continue
        pieces = []
        if item.get("type") == "text":
            text = item.get("plain", "").strip()
            if text:
                pieces.append(text)
        elif item.get("type") == "table":
            for row in item.get("rows", []):
                if isinstance(row, (list, tuple)):
                    cells = [str(cell).strip() for cell in row if str(cell).strip()]
                    if cells:
                        pieces.append(" | ".join(cells))
        if pieces:
            pages.setdefault(int(page_num), []).extend(pieces)
    return {page: "\n\n".join(parts) for page, parts in pages.items()}


def _cache_study_content(chat_id, file_name, content_items):
    study_text = _study_source_text(content_items)
    if not study_text:
        return False
    study_pages = _study_page_texts(content_items)
    _study_material_cache[chat_id] = {
        "text": study_text,
        "pages": study_pages,
        "page_count": max(study_pages, default=-1) + 1,
        "file_name": file_name,
        "timestamp": _time.time(),
        "selected_pages": None,
        "quiz_count": 20,
        "difficulty": "medium",
        "answer_key": "separate",
    }
    return True


def _cache_study_from_file(chat_id, file_path, file_name, status=None):
    """Build study source for translation modes that do not expose extracted items."""
    extension = os.path.splitext(file_name)[1].lower().lstrip(".")
    try:
        if extension == "pdf":
            native_items = []
            with fitz.open(file_path) as pdf:
                for page_num, page in enumerate(pdf):
                    text = page.get_text("text").strip()
                    if text:
                        native_items.append({"type": "text", "plain": text, "page_num": page_num})
            if _cache_study_content(chat_id, file_name, native_items):
                return True
            scanned = True
        else:
            scanned = False

        if status:
            status.info("📚 جاري تجهيز نص المحاضرة لخيارات المذاكرة...")
        if extension == "pdf":
            input_file = file_path
            extracted = extract_content(
                file_obj=input_file, file_type=extension, api_key=GEMINI_API_KEY,
                extract_images=False, is_scanned=scanned,
            )
        else:
            with open(file_path, "rb") as input_file:
                extracted = extract_content(
                    file_obj=input_file, file_type=extension, api_key=GEMINI_API_KEY,
                    extract_images=False, is_scanned=False,
                )
        return _cache_study_content(chat_id, file_name, extracted)
    except Exception as exc:
        print(f"Could not prepare study source: {type(exc).__name__}: {exc}")
        return False


def _send_study_options(chat_id):
    bot.send_message(chat_id, "📚 اختر مادة للمذاكرة:", reply_markup=_study_keyboard())


def _study_keyboard():
    markup = telebot.types.InlineKeyboardMarkup()
    markup.add(telebot.types.InlineKeyboardButton("📝 Create English Quiz", callback_data="study_quiz"))
    markup.add(telebot.types.InlineKeyboardButton("🧠 Visual Mind Map", callback_data="study_mind_map"))
    markup.add(telebot.types.InlineKeyboardButton("📚 Study Summary", callback_data="study_summary"))
    markup.add(telebot.types.InlineKeyboardButton("📖 English-Arabic Glossary", callback_data="study_glossary"))
    markup.add(telebot.types.InlineKeyboardButton("📄 Select Pages / Slides", callback_data="study_pages"))
    return markup


def _quiz_settings_keyboard():
    markup = telebot.types.InlineKeyboardMarkup()
    markup.row(*[telebot.types.InlineKeyboardButton(f"{n} questions", callback_data=f"study_count_{n}") for n in (10, 20, 30)])
    markup.add(telebot.types.InlineKeyboardButton("Custom number", callback_data="study_count_custom"))
    return markup


def _quiz_difficulty_keyboard():
    markup = telebot.types.InlineKeyboardMarkup()
    markup.row(
        telebot.types.InlineKeyboardButton("Easy", callback_data="study_difficulty_easy"),
        telebot.types.InlineKeyboardButton("Medium", callback_data="study_difficulty_medium"),
        telebot.types.InlineKeyboardButton("Hard", callback_data="study_difficulty_hard"),
    )
    return markup


def _answer_key_keyboard():
    markup = telebot.types.InlineKeyboardMarkup()
    markup.row(
        telebot.types.InlineKeyboardButton("Separate answer file", callback_data="study_key_separate"),
        telebot.types.InlineKeyboardButton("Include at end", callback_data="study_key_included"),
    )
    return markup

# Rate limiting
from collections import defaultdict
import time as _time
_user_last_request = defaultdict(float)
RATE_LIMIT_SECONDS = 30
MAX_FILE_SIZE_MB = 20
try:
    MAX_PDF_PAGES = max(1, int(os.getenv("MAX_PDF_PAGES", "120")))
except ValueError:
    MAX_PDF_PAGES = 120


def _validate_pdf_page_count(file_path, status):
    try:
        with fitz.open(file_path) as pdf:
            page_count = len(pdf)
    except Exception as exc:
        status.error(f"تعذر فتح ملف PDF: {exc}")
        return False
    if page_count > MAX_PDF_PAGES:
        status.error(
            f"عدد الصفحات ({page_count}) يتجاوز الحد الحالي ({MAX_PDF_PAGES}). "
            "يمكن تعديل MAX_PDF_PAGES في إعدادات التشغيل."
        )
        return False
    return True

def _check_rate_limit(user_id: int) -> bool:
    now = _time.time()
    if now - _user_last_request[user_id] < RATE_LIMIT_SECONDS:
        return False
    _user_last_request[user_id] = now
    return True

def _cache_set(chat_id, document):
    """Store a document in the cache and evict expired entries."""
    now = _time.time()
    user_files_cache[chat_id] = (document, now)
    stale = [k for k, (_, ts) in user_files_cache.items() if now - ts > _CACHE_TTL_SEC]
    for k in stale:
        del user_files_cache[k]

def _cache_pop(chat_id):
    """Remove and return the cached document, or None if missing/expired."""
    entry = user_files_cache.pop(chat_id, None)
    if entry is None:
        return None
    document, ts = entry
    if _time.time() - ts > _CACHE_TTL_SEC:
        return None
    return document


_active_file_jobs = set()
_active_file_jobs_lock = threading.Lock()


def _start_file_job(call, document_obj, target, args):
    """Run at most one translation job for the same uploaded file per chat."""
    chat_id = call.message.chat.id
    file_key = (chat_id, getattr(document_obj, "file_unique_id", None)
                or document_obj.file_id)
    with _active_file_jobs_lock:
        if file_key in _active_file_jobs:
            bot.answer_callback_query(
                call.id,
                "هذا الملف قيد الترجمة بالفعل؛ انتظر النتيجة قبل بدء طلب آخر.",
                show_alert=True,
            )
            return False
        _active_file_jobs.add(file_key)

    try:
        bot.answer_callback_query(call.id)
    except Exception:
        pass

    def _run():
        try:
            target(*args)
        finally:
            with _active_file_jobs_lock:
                _active_file_jobs.discard(file_key)

    threading.Thread(target=_run).start()
    return True


@bot.message_handler(content_types=['photo'])
def handle_photo(message: Message):
    """Translate a photo sent directly (e.g. a page photographed from a book)."""
    if not GEMINI_API_KEY:
        bot.reply_to(message, "مفتاح Gemini API غير متوفر.")
        return
    if not _check_rate_limit(message.from_user.id):
        bot.reply_to(message, f"⏳ يرجى الانتظار {RATE_LIMIT_SECONDS} ثانية بين كل طلب.")
        return

    status_msg = bot.reply_to(message, "⏳ جاري تحليل وترجمة الصورة...")
    threading.Thread(target=_process_photo, args=(message, status_msg.message_id)).start()


def _process_photo(message, status_message_id):
    import traceback
    status = TelegramStatusMock(bot, message.chat.id, "⏳ جاري تحميل الصورة...", message_id=status_message_id)
    try:
        photo = message.photo[-1]  # Highest resolution
        image_bytes = _download_telegram_file(
            photo.file_id, status=status, expected_size=getattr(photo, "file_size", None),
        )

        status.info("🔍 تحليل النص وترجمته بالذكاء الاصطناعي...")
        items = extract_and_translate_page(image_bytes, GEMINI_API_KEY)

        if not items:
            status.error("لم يتم العثور على نص في الصورة.")
            return

        # Build a readable text message
        result_parts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get('type') == 'text':
                en = item.get('english', '').strip()
                ar = item.get('arabic', '').strip()
                if en or ar:
                    result_parts.append(f"📖 {en}\n🔹 {ar}")

        if result_parts:
            full_text = "\n\n".join(result_parts)
            # Telegram message limit is 4096 chars
            if len(full_text) > 4000:
                full_text = full_text[:4000] + "\n\n... (تم اقتطاع النص)"
            status.success("✅ تمت الترجمة!")
            bot.send_message(message.chat.id, full_text)
        else:
            status.error("لم يتم العثور على نص قابل للترجمة.")
    except Exception as e:
        print(traceback.format_exc())
        status.error(f"حدث خطأ: {str(e)}")


@bot.message_handler(content_types=['document'])
def handle_document(message: Message):
    if not message.document:
        return

    file_name = message.document.file_name.lower()
    if not (file_name.endswith('.pdf') or file_name.endswith('.pptx')):
        bot.reply_to(message, "عذراً، أستقبل فقط ملفات بصيغة PDF أو PPTX.\n\n💡 يمكنك أيضاً إرسال صورة مباشرة لترجمتها!")
        return

    # File size check
    if message.document.file_size and message.document.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        bot.reply_to(message, f"⚠️ حجم الملف يتجاوز الحد الأقصى ({MAX_FILE_SIZE_MB}MB).")
        return

    # Rate limiting
    if not _check_rate_limit(message.from_user.id):
        bot.reply_to(message, f"⏳ يرجى الانتظار {RATE_LIMIT_SECONDS} ثانية بين كل طلب.")
        return

    _study_material_cache.pop(message.chat.id, None)

    # Store file in cache (with TTL)
    _cache_set(message.chat.id, message.document)

    # Processing mode buttons
    markup = telebot.types.InlineKeyboardMarkup()
    btn_smart = telebot.types.InlineKeyboardButton("✨ ثنائية بالتصميم الأصلي (صندوق عربي تحت كل فقرة) ⭐", callback_data="opt_smart_boxes")
    btn_transparent = telebot.types.InlineKeyboardButton("🔴 ترجمة فوق الأصل مع ترحيل تلقائي للصفحات", callback_data="opt_transparent_overlay")
    btn0 = telebot.types.InlineKeyboardButton("🧹 استبدال الإنجليزية بالعربية (نفس المكان)", callback_data="opt_overlay")
    btn1 = telebot.types.InlineKeyboardButton("📝 استخراج نصي (بجانب النص - عمودين)",    callback_data="opt_auto_side")
    btn2 = telebot.types.InlineKeyboardButton("📝 استخراج نصي (تحت النص - Word/PDF)",      callback_data="opt_auto_stacked")
    btn3 = telebot.types.InlineKeyboardButton("🔲 سلايدات — مقارنة جانبية",    callback_data="opt_slides_side")
    btn4 = telebot.types.InlineKeyboardButton("🌙 سلايدات — عربي كامل الحجم", callback_data="opt_slides_arabic")

    markup.add(btn_smart)
    markup.add(btn_transparent)
    markup.add(btn0)
    markup.add(btn1)
    markup.add(btn2)
    markup.add(btn3)
    markup.add(btn4)

    bot.reply_to(
        message,
        "✅ تم استلام الملف بنجاح!\n\n"
        "اختر وضع المعالجة:\n"
        "✨ *ثنائية بالتصميم الأصلي* ⭐ — يحتفظ بتصميم وخلفيات الملف الأصلية 100% ويضع صندوق ترجمة عربي ذكي تحت كل فقرة\n"
        "🔴 *فوق الأصل مع ترحيل الصفحات* — يبقي النص الأصلي ويضيف العربية، وينقل المحتوى المتبقي إلى صفحات متابعة عند الحاجة\n"
        "🧹 *استبدال النص* — يستبدل الإنجليزية بالعربية في نفس المكان\n"
        "📝 *استخراج نصي (تحت النص)* — يستخرج النص كملف وورد و PDF جديد\n"
        "📝 *استخراج نصي (بجانب النص)* — يستخرج النص ويعيد تصميمه كعمودين متقابلين\n"
        "🔲 *سلايدات جانبية* — مقارنة الشريحة الأصلية والمترجمة جنبًا إلى جنب\n"
        "🌙 *سلايدات كاملة* — شريحة عربية بحجم كامل (أسهل للقراءة)\n\n"
        "💡 كشف نوع الملف (نصي/مصوّر) يتم تلقائياً.",
        reply_markup=markup,
        parse_mode='Markdown'
    )

    # Notify Bot Owner (Optional)
    if BOT_OWNER_ID:
        try:
            user = message.from_user
            user_info = f"👤 مستخدم: {user.first_name}"
            if user.last_name: user_info += f" {user.last_name}"
            if user.username: user_info += f" (@{user.username})"
            user_info += f"\n🆔 ID: `{user.id}`"
            
            notification = (
                f"🔔 **تنبيه استخدام جديد**\n\n"
                f"{user_info}\n"
                f"📄 الملف: `{message.document.file_name}`\n"
                f"📦 النوع: `{message.document.mime_type}`"
            )
            bot.send_message(BOT_OWNER_ID, notification, parse_mode='Markdown')
        except Exception as e:
            print(f"Error sending owner notification: {e}")

@bot.callback_query_handler(func=lambda call: call.data.startswith("study_"))
def handle_study_material(call):
    chat_id = call.message.chat.id
    entry = _study_material_cache.get(chat_id)
    if not entry or _time.time() - entry["timestamp"] > _STUDY_CACHE_TTL_SEC:
        _study_material_cache.pop(chat_id, None)
        bot.answer_callback_query(call.id, "انتهت صلاحية محتوى المحاضرة؛ أعد إرسال الملف وترجمته أولاً.", show_alert=True)
        return

    try:
        bot.answer_callback_query(call.id)
    except Exception:
        pass
    action = call.data.removeprefix("study_")
    if action == "quiz":
        bot.send_message(chat_id, "Choose the number of quiz questions:", reply_markup=_quiz_settings_keyboard())
    elif action.startswith("count_"):
        choice = action.removeprefix("count_")
        if choice == "custom":
            prompt = bot.send_message(chat_id, "أرسل عدد الأسئلة (من 5 إلى 50).")
            bot.register_next_step_handler(prompt, _accept_custom_quiz_count, chat_id)
        else:
            entry["quiz_count"] = int(choice)
            bot.send_message(chat_id, "Choose quiz difficulty:", reply_markup=_quiz_difficulty_keyboard())
    elif action.startswith("difficulty_"):
        entry["difficulty"] = action.removeprefix("difficulty_")
        bot.send_message(chat_id, "How should the answer key be delivered?", reply_markup=_answer_key_keyboard())
    elif action.startswith("key_"):
        entry["answer_key"] = action.removeprefix("key_")
        _start_study_job(chat_id, entry, "quiz")
    elif action in {"summary", "glossary", "mind_map"}:
        _start_study_job(chat_id, entry, action)
    elif action == "pages":
        page_count = entry.get("page_count", 0)
        if not entry.get("pages"):
            bot.send_message(chat_id, "تعذر تحديد الصفحات في هذا الملف؛ استخدم خيارات الدراسة على كامل المحاضرة.")
            return
        prompt = bot.send_message(
            chat_id,
            f"أرسل أرقام الصفحات/الشرائح (مثلاً 1-3, 5) أو فصلاً (مثلاً chapter 2). "
            f"عدد الصفحات/الشرائح ذات النص: {page_count}.\nأرسل all لاستخدام المحاضرة كاملة.",
        )
        bot.register_next_step_handler(prompt, _accept_study_page_range, chat_id)


def _accept_custom_quiz_count(message, chat_id):
    entry = _study_material_cache.get(chat_id)
    try:
        count = int((message.text or "").strip())
    except ValueError:
        bot.reply_to(message, "أرسل رقماً صحيحاً بين 5 و50.")
        return
    if not entry or not 5 <= count <= 50:
        bot.reply_to(message, "عدد الأسئلة يجب أن يكون من 5 إلى 50. ابدأ اختيار الاختبار مجدداً.")
        return
    entry["quiz_count"] = count
    bot.send_message(chat_id, "Choose quiz difficulty:", reply_markup=_quiz_difficulty_keyboard())


def _accept_study_page_range(message, chat_id):
    entry = _study_material_cache.get(chat_id)
    if not entry:
        bot.reply_to(message, "انتهت صلاحية محتوى المحاضرة؛ أعد إرسال الملف وترجمته.")
        return
    raw = (message.text or "").strip().lower()
    if raw == "all":
        entry["selected_pages"] = None
        bot.reply_to(message, "تم اختيار المحاضرة كاملة.", reply_markup=_study_keyboard())
        return
    selected = set()
    try:
        chapter_tokens = re.findall(r"(?:chapter|unit|lecture)\s+(\d+)", raw)
        if chapter_tokens:
            page_starts = []
            for page, page_text in sorted(entry["pages"].items()):
                match = re.search(r"\b(?:chapter|unit|lecture)\s+(\d+)\b", page_text, re.IGNORECASE)
                if match:
                    page_starts.append((int(match.group(1)), page))
            for chapter in map(int, chapter_tokens):
                starts = [(number, page) for number, page in page_starts if number == chapter]
                if not starts:
                    raise ValueError
                start_page = starts[0][1]
                later = [page for _, page in page_starts if page > start_page]
                end_page = min(later) if later else entry["page_count"]
                selected.update(page for page in entry["pages"] if start_page <= page < end_page)
        else:
            for token in raw.split(","):
                token = token.strip()
                if "-" in token:
                    start, end = (int(part.strip()) for part in token.split("-", 1))
                    if start > end:
                        raise ValueError
                    selected.update(range(start - 1, end))
                else:
                    selected.add(int(token) - 1)
    except ValueError:
        bot.reply_to(message, "صيغة غير واضحة. أرسل أرقاماً مثل 1-3, 5 أو all.")
        return
    available = set(entry["pages"])
    if not selected or not selected.issubset(available):
        bot.reply_to(message, "تأكد من اختيار صفحات تحتوي على نص، ضمن مدى الصفحات المتاح.")
        return
    entry["selected_pages"] = sorted(selected)
    bot.reply_to(
        message,
        "تم تحديد الصفحات. اختر نوع المادة الدراسية:",
        reply_markup=_study_keyboard(),
    )


def _study_text_for_entry(entry):
    selected = entry.get("selected_pages")
    if selected is None:
        return entry["text"]
    return "\n\n".join(
        f"[Page/Slide {page + 1}]\n{entry['pages'][page]}"
        for page in selected if page in entry["pages"]
    )[:60000]


def _send_text_file(chat_id, name, content, caption):
    output = io.BytesIO(content.encode("utf-8"))
    output.name = name
    bot.send_document(chat_id, output, caption=caption)


def _render_mind_map_png(outline):
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    lines = [line for line in outline.splitlines() if line.strip()][:70]
    if not lines:
        raise ValueError("The generated mind map was empty")
    font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
    title_font = ImageFont.truetype(font_path, 36)
    body_font = ImageFont.truetype(font_path, 23)
    prepared = []
    for line in lines:
        stripped = line.strip().lstrip("#*-• ").strip()
        indent = len(line) - len(line.lstrip(" "))
        level = min(3, indent // 2)
        wrapped = textwrap.wrap(stripped, width=max(24, 52 - level * 4)) or [stripped]
        prepared.append((level, wrapped))
    row_heights = [max(82, 34 + 32 * len(wrapped)) for _, wrapped in prepared]
    height = min(8200, 130 + sum(row_heights) + 14 * len(row_heights))
    image = Image.new("RGB", (1200, height), "#F5F7FB")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 24, 1160, 92), radius=18, fill="#183B56")
    draw.text((68, 37), "LECTURE MIND MAP", font=title_font, fill="white")
    colors = ["#1F6E8C", "#4F8A8B", "#7A9E7E", "#A5A58D"]
    parent_centers = {}
    y = 118
    for (level, wrapped), row_height in zip(prepared, row_heights):
        x = 55 + level * 72
        box_right = 1140 - level * 20
        box_bottom = y + row_height - 10
        color = colors[level]
        draw.rounded_rectangle((x, y, box_right, box_bottom), radius=15, fill="white", outline=color, width=4)
        center_y = (y + box_bottom) // 2
        parent = parent_centers.get(level - 1)
        if level and parent:
            draw.line((x - 20, parent, x - 20, center_y), fill=color, width=4)
            draw.line((x - 20, center_y, x, center_y), fill=color, width=4)
        parent_centers[level] = center_y
        for i, text_line in enumerate(wrapped):
            draw.text((x + 20, y + 17 + 31 * i), text_line, font=body_font, fill="#172B4D")
        for old_level in list(parent_centers):
            if old_level > level:
                del parent_centers[old_level]
        y += row_height + 14
        if y + 100 > height:
            break
    return image


def _start_study_job(chat_id, entry, kind):
    with _active_study_jobs_lock:
        if chat_id in _active_study_jobs:
            bot.send_message(chat_id, "يجري إعداد مادة دراسية لهذا الملف بالفعل.")
            return
        _active_study_jobs.add(chat_id)
    labels = {"quiz": "the English quiz", "mind_map": "the mind map",
              "summary": "the study summary", "glossary": "the glossary"}
    status_msg = bot.send_message(chat_id, f"⏳ Preparing {labels[kind]}...")

    def _run():
        try:
            source = _study_text_for_entry(entry)
            result = generate_study_material(
                source, kind, GEMINI_API_KEY,
                question_count=entry.get("quiz_count", 20),
                difficulty=entry.get("difficulty", "medium"),
                answer_key=entry.get("answer_key", "separate"),
            )
            safe_name = os.path.splitext(os.path.basename(entry["file_name"]))[0]
            if kind == "quiz":
                parts = re.split(r"(?m)^=== ANSWER KEY ===\s*$", result, maxsplit=1)
                questions = re.sub(r"(?m)^=== QUESTIONS ===\s*", "", parts[0]).strip()
                if len(parts) > 1:
                    answer_key = parts[1].strip()
                    if entry.get("answer_key", "separate") == "separate":
                        _send_text_file(chat_id, f"{safe_name}_quiz.txt", questions, "English practice test")
                        _send_text_file(chat_id, f"{safe_name}_answer_key.txt", answer_key, "Answer key — keep separate while practicing")
                    else:
                        _send_text_file(chat_id, f"{safe_name}_quiz.txt", questions + "\n\nANSWER KEY\n" + answer_key, "English practice test with answer key")
                else:
                    _send_text_file(chat_id, f"{safe_name}_quiz.txt", questions, "English practice test")
            elif kind == "mind_map":
                png = _render_mind_map_png(result)
                buffer = io.BytesIO()
                png.save(buffer, format="PNG")
                buffer.seek(0)
                buffer.name = f"{safe_name}_mind_map.png"
                bot.send_photo(chat_id, buffer, caption="Visual mind map (English)")
                _send_text_file(chat_id, f"{safe_name}_mind_map.txt", result, "Editable mind map outline")
            else:
                _send_text_file(chat_id, f"{safe_name}_{kind}.txt", result,
                                "English study summary" if kind == "summary" else "English-Arabic glossary")
            bot.delete_message(chat_id, status_msg.message_id)
        except Exception as exc:
            print(f"Study material generation failed: {type(exc).__name__}: {exc}")
            bot.edit_message_text("تعذر إعداد المادة الدراسية الآن. حاول مرة أخرى لاحقاً.",
                                  chat_id=chat_id, message_id=status_msg.message_id)
        finally:
            with _active_study_jobs_lock:
                _active_study_jobs.discard(chat_id)

    threading.Thread(target=_run, daemon=True).start()


@bot.callback_query_handler(func=lambda call: call.data.startswith("opt_"))
def handle_options(call):
    chat_id = call.message.chat.id
    document_obj = _cache_pop(chat_id)
    if document_obj is None:
        bot.answer_callback_query(call.id, "عذراً، انتهت صلاحية الملف السابقة، يرجى إعادة إرساله.", show_alert=True)
        return
    file_name = document_obj.file_name.lower()
    
    # Update the keyboard message to act as status message
    bot.edit_message_text("⏳ جاري تجهيز الملف...", chat_id=chat_id, message_id=call.message.message_id)



    # ── Smart Bilingual on Original Design ──
    if call.data in ("opt_smart_boxes", "opt_bilingual", "opt_transparent_overlay"):
        if not file_name.endswith('.pdf'):
            bot.edit_message_text("⚠️ هذا الوضع يدعم ملفات PDF فقط حالياً.",
                                 chat_id=chat_id, message_id=call.message.message_id)
            return
        _start_file_job(
            call, document_obj, process_smart_bilingual_mode,
            (chat_id, document_obj, file_name, call.message.message_id,
             call.data == "opt_transparent_overlay"),
        )
        return

    # ── Overlay mode ──
    if call.data == "opt_overlay":
        if not file_name.endswith('.pdf'):
            bot.edit_message_text("⚠️ وضع الترجمة الشفافة يدعم ملفات PDF فقط حالياً.",
                                 chat_id=chat_id, message_id=call.message.message_id)
            return
        _start_file_job(
            call, document_obj, process_overlay_mode,
            (chat_id, document_obj, file_name, call.message.message_id),
        )
        return

    # ── Slide translation modes ──
    if call.data in ("opt_slides_side", "opt_slides_arabic", "opt_slides"):
        layout = 'arabic_only' if call.data == "opt_slides_arabic" else 'side_by_side'
        _start_file_job(
            call, document_obj, process_slides_mode,
            (chat_id, document_obj, file_name, call.message.message_id, layout),
        )
        return

    # Auto-detect scanned vs text PDF (replaces manual ocr button)
    is_scanned = False  # will be auto-detected in process_uploaded_file
    stacked = "stacked" in call.data
    auto_detect = "auto" in call.data

    _start_file_job(
        call, document_obj, process_uploaded_file,
        (chat_id, document_obj, file_name, call.message.message_id,
         is_scanned, stacked, auto_detect),
    )

def process_uploaded_file(chat_id, document_obj, file_name, status_message_id, is_scanned, stacked, auto_detect=False):
    import traceback
    start_time = _time.time()
    status = TelegramStatusMock(bot, chat_id, "⏳ جاري تحميل الملف من خوادم تليكرام...", message_id=status_message_id)
    progress = TelegramProgressMock(bot, chat_id, status.message_id)
    
    file_extension = file_name.split('.')[-1]
    temp_file_path = _new_temp_file_path(file_name)
    pdf_path = docx_path = None
    
    try:
        if not GEMINI_API_KEY:
            status.error("عذراً، مفتاح Gemini API غير متوفر في إعدادات البوت.")
            return

        # Download file
        downloaded_file = _download_telegram_file(
            document_obj.file_id, status=status,
            expected_size=getattr(document_obj, "file_size", None),
        )
        
        with open(temp_file_path, "wb") as f:
            f.write(downloaded_file)

        if file_extension == 'pdf' and not _validate_pdf_page_count(temp_file_path, status):
            return

        # Auto-detect scanned PDF if requested
        if auto_detect and file_extension == 'pdf':
            status.info("🔍 جاري كشف نوع الملف تلقائياً...")
            is_scanned = is_scanned_pdf(temp_file_path)
            mode_label = "مصوّر (OCR)" if is_scanned else "نصي"
            status.info(f"📋 نوع الملف: {mode_label} — الخطوة 1/3: استخراج المحتوى...")
        else:
            status.info("الخطوة 1/3: استخراج النصوص والصور من الملف...")

        # Open file as binary stream for pdf_processor
        with open(temp_file_path, "rb") as file_stream:
            # We recreate a minimal object that has a ".name" and behaves like Streamlit's UploadedFile
            class MockFile:
                def __init__(self, name, stream):
                    self.name = name
                    self.stream = stream
                def read(self): return self.stream.read()
                def seek(self, pos): return self.stream.seek(pos)
                
            mock_uploaded_file = MockFile(file_name, file_stream)
            
            # Since pdf_processor's extract_content checks `isinstance(pdf_file, str)` or uses stream
            # We pass the string path for PDF, mock file for PPTX to be totally safe
            input_file = temp_file_path if file_extension == 'pdf' else mock_uploaded_file
            
            content_items = extract_content(
                file_obj=input_file,
                file_type=file_extension,
                api_key=GEMINI_API_KEY,
                extract_images=True,
                is_scanned=is_scanned,
                status_text=status,
                progress_bar=progress
            )
            
            if not content_items:
                status.error("لم يتم العثور على محتوى قابل للترجمة في هذا الملف.")
                return

            status.info(f"الخطوة 2/3: ترجمة المحتوى ({len(content_items)} عناصر)...")
            integrate_translations(content_items, GEMINI_API_KEY)

            status.info("الخطوة 3/3: تصميم وتوليد ملفات PDF و Word النهائية...")
            pdf_path, docx_path = generate_documents(content_items, stacked=stacked)
            
            if pdf_path and docx_path:
                # Compute stats
                elapsed = _time.time() - start_time
                text_count = sum(1 for it in content_items if it['type'] == 'text')
                table_count = sum(1 for it in content_items if it['type'] == 'table')
                image_count = sum(1 for it in content_items if it['type'] == 'image')
                mode_str = "OCR" if is_scanned else "نصي"
                
                status.success("🎉 تمت الترجمة بنجاح! جاري الإرسال...")
                
                # Construct meaningful output filenames
                base_name = os.path.splitext(file_name)[0]
                pdf_output_name = f"{base_name}_translated.pdf"
                docx_output_name = f"{base_name}_translated.docx"
                
                with open(pdf_path, 'rb') as f_pdf:
                    bot.send_document(chat_id, (pdf_output_name, f_pdf), caption="النسخة المترجمة (PDF)")

                with open(docx_path, 'rb') as f_docx:
                    bot.send_document(chat_id, (docx_output_name, f_docx), caption="النسخة المترجمة (Word)")

                # Clean up generated output files
                for out_path in (pdf_path, docx_path):
                    try:
                        if out_path and os.path.exists(out_path):
                            os.remove(out_path)
                    except Exception:
                        pass

                # Send stats summary
                stats_msg = (
                    f"📊 **ملخص الترجمة**\n\n"
                    f"📋 الوضع: {mode_str}\n"
                    f"📝 فقرات: {text_count}\n"
                    f"📋 جداول: {table_count}\n"
                    f"🖼️ صور: {image_count}\n"
                    f"⏱️ الوقت: {elapsed:.1f} ثانية"
                )
                if _cache_study_content(chat_id, file_name, content_items):
                    bot.send_message(
                        chat_id,
                        stats_msg + "\n\n📚 يمكنك الآن إنشاء مواد للمذاكرة:",
                        parse_mode='Markdown',
                        reply_markup=_study_keyboard(),
                    )
                else:
                    bot.send_message(chat_id, stats_msg, parse_mode='Markdown')
            else:
                status.error("حدث خطأ أثناء محاولة إنشاء الملفات النهائية.")
                
    except GeminiQuotaExceeded as e:
        print("[GEMINI QUOTA] Quota or request limit reached; translation stopped.")
        status.error(str(e))
    except Exception as e:
        print(traceback.format_exc())
        status.error(f"عذراً، حدث خطأ غير متوقع: {str(e)}")
        try:
            bot.send_message(chat_id, f"❌ خطأ تقني: {str(e)[:500]}")
        except Exception:
            pass
    finally:
        # Clean up temp file
        _cleanup_temp_files(temp_file_path, pdf_path, docx_path)


def process_slides_mode(chat_id, document_obj, file_name, status_message_id, layout='side_by_side'):
    """Handles the Slide Translation mode end-to-end in a background thread."""
    import traceback
    status = TelegramStatusMock(bot, chat_id, "⏳ جاري تحميل الملف من خوادم تليكرام...", message_id=status_message_id)
    progress = TelegramProgressMock(bot, chat_id, status.message_id)

    file_extension = file_name.split('.')[-1]
    temp_file_path = _new_temp_file_path(file_name)
    output_pdf_path = None

    try:
        if not GEMINI_API_KEY:
            status.error("عذراً، مفتاح Gemini API غير متوفر في إعدادات البوت.")
            return

        # ── Only PDF is supported in slides mode ──
        if file_extension != 'pdf':
            status.error(
                "⚠️ وضع السلايدات يدعم ملفات PDF فقط في الوقت الحالي.\n"
                "يرجى تحويل ملف PPTX إلى PDF أولاً ثم إعادة إرساله."
            )
            return

        # Download
        downloaded_file = _download_telegram_file(
            document_obj.file_id, status=status,
            expected_size=getattr(document_obj, "file_size", None),
        )
        with open(temp_file_path, "wb") as f:
            f.write(downloaded_file)

        if not _validate_pdf_page_count(temp_file_path, status):
            return

        status.info("🎨 وضع السلايدات: تحليل الشرائح وترجمتها...")

        output_pdf_path = process_as_slides(
            file_obj=temp_file_path,
            file_type='pdf',
            api_key=GEMINI_API_KEY,
            status_text=status,
            progress_bar=progress,
            layout=layout,
        )

        if output_pdf_path:
            has_study_materials = _cache_study_from_file(chat_id, temp_file_path, file_name, status)
            status.success("🎉 تمت ترجمة السلايدات بنجاح! جاري الإرسال...")
            base_name = os.path.splitext(file_name)[0]
            layout_tag = "arabic_only" if layout == 'arabic_only' else "bilingual"
            out_name = f"{base_name}_slides_{layout_tag}.pdf"
            caption = (
                "✅ الشرائح المترجمة — عربي كامل الحجم"
                if layout == 'arabic_only'
                else "✅ الشرائح المترجمة — مقارنة جانبية (أصل | ترجمة)"
            )
            with open(output_pdf_path, 'rb') as f_out:
                bot.send_document(chat_id, (out_name, f_out), caption=caption)
            if has_study_materials:
                _send_study_options(chat_id)
            # Cleanup output
            try:
                os.remove(output_pdf_path)
            except Exception:
                pass
        else:
            status.error("حدث خطأ أثناء معالجة السلايدات. يرجى المحاولة مجدداً.")

    except GeminiQuotaExceeded as e:
        print("[GEMINI QUOTA] Quota or request limit reached; translation stopped.")
        status.error(str(e))
    except Exception as e:
        print(traceback.format_exc())
        status.error(f"عذراً، حدث خطأ غير متوقع: {str(e)}")
    finally:
        _cleanup_temp_files(temp_file_path, output_pdf_path)


def process_overlay_mode(chat_id, document_obj, file_name, status_message_id):
    """Handles the Overlay Translation mode — preserves original PDF layout."""
    import traceback
    start_time = _time.time()
    status = TelegramStatusMock(bot, chat_id, "⏳ جاري تحميل الملف...", message_id=status_message_id)
    progress = TelegramProgressMock(bot, chat_id, status.message_id)

    temp_file_path = _new_temp_file_path(file_name)
    output_path = None

    try:
        if not GEMINI_API_KEY:
            status.error("عذراً، مفتاح Gemini API غير متوفر.")
            return

        # Download
        downloaded_file = _download_telegram_file(
            document_obj.file_id, status=status,
            expected_size=getattr(document_obj, "file_size", None),
        )
        with open(temp_file_path, "wb") as f:
            f.write(downloaded_file)

        if not _validate_pdf_page_count(temp_file_path, status):
            return

        status.info("📋 وضع الترجمة الشفافة — يحافظ على التنسيق الأصلي...")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_out:
            output_path = tmp_out.name

        success = generate_overlay_pdf(
            pdf_path=temp_file_path,
            api_key=GEMINI_API_KEY,
            output_path=output_path,
            status_text=status,
            progress_bar=progress
        )

        if success and os.path.exists(output_path):
            has_study_materials = _cache_study_from_file(chat_id, temp_file_path, file_name, status)
            elapsed = _time.time() - start_time
            status.success("🎉 تمت الترجمة الشفافة بنجاح! جاري الإرسال...")

            base_name = os.path.splitext(file_name)[0]
            out_name = f"{base_name}_overlay.pdf"

            with open(output_path, 'rb') as f_out:
                bot.send_document(chat_id, (out_name, f_out),
                                  caption="✅ الترجمة الشفافة — التنسيق الأصلي محفوظ 100%")

            bot.send_message(chat_id,
                f"📊 **ملخص**\n⏱️ الوقت: {elapsed:.1f} ثانية\n"
                f"📋 الوضع: ترجمة شفافة (Overlay)",
                parse_mode='Markdown')
            if has_study_materials:
                _send_study_options(chat_id)

            try:
                os.remove(output_path)
            except Exception:
                pass
        else:
            status.error("حدث خطأ أثناء إنشاء الملف المترجم.")

    except GeminiQuotaExceeded as e:
        print("[GEMINI QUOTA] Quota or request limit reached; translation stopped.")
        status.error(str(e))
    except Exception as e:
        print(traceback.format_exc())
        status.error(f"عذراً، حدث خطأ: {str(e)}")
    finally:
        _cleanup_temp_files(temp_file_path, output_path)


def process_smart_bilingual_mode(chat_id, document_obj, file_name, status_message_id,
                                 transparent=False):
    """Handles the Smart Bilingual mode — original layout with Arabic box below each paragraph."""
    import traceback
    start_time = _time.time()
    status = TelegramStatusMock(bot, chat_id, "⏳ جاري تحميل الملف...", message_id=status_message_id)
    progress = TelegramProgressMock(bot, chat_id, status.message_id)

    temp_file_path = _new_temp_file_path(file_name)
    output_path = None

    try:
        # Download
        downloaded_file = _download_telegram_file(
            document_obj.file_id, status=status,
            expected_size=getattr(document_obj, "file_size", None),
        )
        with open(temp_file_path, "wb") as f:
            f.write(downloaded_file)

        if not _validate_pdf_page_count(temp_file_path, status):
            return

        if not GEMINI_API_KEY:
            status.error("عذراً، مفتاح Gemini API غير متوفر.")
            return

        if transparent:
            status.info("🔴 ترجمة فوق الأصل — إدراج العربية مع ترحيل المحتوى عند امتلاء الصفحة...")
        else:
            status.info("✨ ترجمة ثنائية بالتصميم الأصلي — جاري إدراج صناديق الترجمة...")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_out:
            output_path = tmp_out.name

        success = generate_bilingual_overlay_pdf(
            pdf_path=temp_file_path,
            api_key=GEMINI_API_KEY,
            output_path=output_path,
            status_text=status,
            progress_bar=progress,
            transparent=transparent
        )

        if success and os.path.exists(output_path):
            has_study_materials = _cache_study_from_file(chat_id, temp_file_path, file_name, status)
            elapsed = _time.time() - start_time
            status.success("🎉 اكتملت الترجمة الشفافة بنجاح! جاري الإرسال..." if transparent else "🎉 اكتملت الترجمة بنجاح! جاري الإرسال...")

            base_name = os.path.splitext(file_name)[0]
            out_name = (f"{base_name}_transparent_overlay.pdf" if transparent
                        else f"{base_name}_bilingual_original_design.pdf")

            with open(output_path, 'rb') as f_out:
                bot.send_document(chat_id, (out_name, f_out),
                                  caption=("🔴 الترجمة الشفافة — النص الأصلي محفوظ والترجمة العربية باللون الأحمر"
                                           if transparent else "✨ الترجمة الثنائية الذكية — التصميم الأصلي محفوظ مع صندوق ترجمة تحت كل فقرة"))

            bot.send_message(chat_id,
                f"📊 **ملخص**\n⏱️ الوقت: {elapsed:.1f} ثانية\n"
                f"{'🔴 الوضع: ترجمة شفافة فوق الملف' if transparent else '✨ الوضع: ثنائية ذكية بالتصميم الأصلي (Smart Box)'}",
                parse_mode='Markdown')
            if has_study_materials:
                _send_study_options(chat_id)

            try:
                os.remove(output_path)
            except Exception:
                pass
        else:
            status.error("حدث خطأ أثناء إنشاء الملف المترجم.")

    except GeminiQuotaExceeded as e:
        print("[GEMINI QUOTA] Quota or request limit reached; translation stopped.")
        status.error(str(e))
    except Exception as e:
        print(traceback.format_exc())
        status.error(f"عذراً، حدث خطأ: {str(e)}")
    finally:
        _cleanup_temp_files(temp_file_path, output_path)





if __name__ == '__main__':
    import sys
    # Fix Unicode output on Windows terminals that default to cp1252
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from keep_alive import keep_alive
    keep_alive()
    print("[BOT] Telegram Bot is starting...")

    # Clear any previous webhooks or long-polling sessions
    try:
        bot.delete_webhook(drop_pending_updates=True)
    except Exception:
        pass
        
    # إعدادات أمان إضافية لمنع انقطاع الاتصال (Read Timeout) بسبب بطء الشبكة
    apihelper.READ_TIMEOUT = 90
    apihelper.CONNECT_TIMEOUT = 90
    
    # Pending updates were already dropped by delete_webhook above. Keeping
    # skip_pending=False also lets infinity_polling catch transient API timeouts
    # instead of making an unguarded getUpdates call before its retry loop.
    bot.infinity_polling(timeout=60, long_polling_timeout=45, skip_pending=False)
