import concurrent.futures
import threading
from google import genai
from google.genai import types
import logging
from typing import List, Tuple, Generator
import config
import time
import re
import json
import os
import sqlite3
import hashlib
from pydantic import BaseModel, Field

# Setup logging
logger = logging.getLogger(__name__)

_TRANSLATION_PROMPT_VERSION = "academic-en-ar-v1"


def _normalize_translation_text(text: str) -> str:
    """Normalize whitespace/case only, so repeated PDF line wraps match safely."""
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _translation_cache_key(text: str, context_summary: str, model_name: str) -> str:
    material = json.dumps(
        [
            _TRANSLATION_PROMPT_VERSION,
            model_name or config.MODEL_NAME,
            _normalize_translation_text(context_summary),
            _normalize_translation_text(text),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _is_cacheable_translation(text: str) -> bool:
    text = str(text or "").strip()
    return (
        bool(text)
        and not text.startswith("Error")
        and not text.startswith("[Error")
        and not text.startswith("[Translation unavailable")
        and not text.startswith("[تعذرت")
        and "تعذرت ترجمة" not in text
        and "GeminiQuotaExceeded" not in text
        and "QuotaExceeded" not in text
        and not text.startswith("[طھ")
    )


def _translation_cache_connection():
    cache_path = getattr(config, "TRANSLATION_CACHE_PATH", "translation_cache.sqlite3")
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    connection = sqlite3.connect(cache_path, timeout=15)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS translations ("
        "cache_key TEXT PRIMARY KEY, translated_text TEXT NOT NULL, accessed_at REAL NOT NULL)"
    )
    return connection


def _load_cached_translations(cache_keys: List[str]) -> dict:
    if not getattr(config, "ENABLE_TRANSLATION_CACHE", True) or not cache_keys:
        return {}
    try:
        with _translation_cache_connection() as connection:
            found = {}
            # Stay below SQLite's parameter limit for unusually large documents.
            for start in range(0, len(cache_keys), 900):
                keys = cache_keys[start:start + 900]
                placeholders = ",".join("?" for _ in keys)
                rows = connection.execute(
                    f"SELECT cache_key, translated_text FROM translations "
                    f"WHERE cache_key IN ({placeholders})",
                    keys,
                ).fetchall()
                for key, trans in rows:
                    if _is_cacheable_translation(trans):
                        found[key] = trans
            if found:
                now = time.time()
                connection.executemany(
                    "UPDATE translations SET accessed_at = ? WHERE cache_key = ?",
                    [(now, key) for key in found],
                )
            return found
    except Exception as exc:
        logger.warning("Translation cache read failed; translating without cache: %s", exc)
        return {}


def _save_cached_translations(entries: List[Tuple[str, str]]) -> None:
    if not getattr(config, "ENABLE_TRANSLATION_CACHE", True) or not entries:
        return
    try:
        now = time.time()
        with _translation_cache_connection() as connection:
            connection.executemany(
                "INSERT INTO translations(cache_key, translated_text, accessed_at) "
                "VALUES (?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET "
                "translated_text = excluded.translated_text, accessed_at = excluded.accessed_at",
                [(key, translated, now) for key, translated in entries if translated.strip()],
            )
            max_entries = max(1, int(getattr(config, "TRANSLATION_CACHE_MAX_ENTRIES", 50000)))
            connection.execute(
                "DELETE FROM translations WHERE cache_key IN ("
                "SELECT cache_key FROM translations ORDER BY accessed_at DESC "
                "LIMIT -1 OFFSET ?)" ,
                (max_entries,),
            )
    except Exception as exc:
        logger.warning("Translation cache write failed; continuing without cache: %s", exc)


# ====== Reusable Client Cache (thread-safe) ======
_client_cache = {}
_client_lock = threading.Lock()
_gemini_request_semaphore = threading.BoundedSemaphore(
    max(1, getattr(config, "MAX_TRANSLATION_WORKERS", 1))
)
_last_request_time = 0.0
_rate_limit_lock = threading.Lock()


def _generate_content(client, **kwargs):
    """Cap concurrent Gemini calls and pace them strictly to stay under Free Tier 5 RPM."""
    global _last_request_time
    min_interval = getattr(config, "MIN_REQUEST_INTERVAL_SECONDS", 13.0)
    with _gemini_request_semaphore:
        with _rate_limit_lock:
            now = time.time()
            elapsed = now - _last_request_time
            if elapsed < min_interval:
                wait_sec = min_interval - elapsed
                logger.info("Pacing request: waiting %.1fs to stay under Free Tier 5 RPM limit...", wait_sec)
                time.sleep(wait_sec)
            _last_request_time = time.time()
        return client.models.generate_content(**kwargs)

def _get_client(api_key: str) -> genai.Client:
    """Returns a cached genai.Client for the given API key (thread-safe)."""
    with _client_lock:
        if api_key not in _client_cache:
            _client_cache[api_key] = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=45_000),
            )
        return _client_cache[api_key]

# ====== Model Fallback Tracker (thread-safe) ======
_active_model = config.MODEL_NAME
_model_lock = threading.Lock()
_fallback_index = 0

def _get_model():
    """Returns the currently active model name (thread-safe)."""
    with _model_lock:
        return _active_model

def _switch_to_fallback(from_model: str = None) -> str:
    """Switch to the next available fallback model when primary is overloaded/exhausted (thread-safe)."""
    global _active_model, _fallback_index
    with _model_lock:
        pool = getattr(config, 'FALLBACK_MODELS', [getattr(config, 'FALLBACK_MODEL_NAME', 'gemini-2.5-flash')])
        all_models = [config.MODEL_NAME] + [m for m in pool if m != config.MODEL_NAME]
        
        curr = from_model or _active_model
        try:
            curr_idx = all_models.index(curr)
            next_idx = (curr_idx + 1) % len(all_models)
        except ValueError:
            next_idx = (_fallback_index + 1) % len(all_models)
            
        next_model = all_models[next_idx]
        _fallback_index = next_idx
        if next_model != _active_model:
            logger.warning(f"⚠️ Switching from {_active_model} to fallback model {next_model}")
            _active_model = next_model
        return _active_model

def _is_quota_error(error: Exception) -> bool:
    """Check if the error is a quota/rate limit exhaustion or server overload (503 UNAVAILABLE / high demand)."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        'resource_exhausted', 'quota', 'rate limit', '429', 
        'too many requests', 'resourceexhausted',
        '503', 'unavailable', 'high demand', 'spikes in demand', 'overloaded'
    ])


def _is_request_limit_error(error: Exception) -> bool:
    """Detect exhausted request/token quotas separately from temporary 503 overloads."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        'resource_exhausted', 'quota', 'rate limit', '429',
        'too many requests', 'resourceexhausted'
    ])

def _is_timeout_error(error: Exception) -> bool:
    """Check if the error is a timeout from the API or local client."""
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        'deadline_exceeded', 'timeout', 'timed out', '504'
    ])


def _is_daily_quota_error(error: Exception) -> bool:
    """Detect exhausted daily quotas, which model fallback cannot fix reliably."""
    error_str = str(error).lower()
    return (
        ('per day' in error_str or 'per_day' in error_str or 'perday' in error_str)
        and any(marker in error_str for marker in ('quota', 'limit exceeded', 'resource_exhausted'))
    )


class GeminiQuotaExceeded(RuntimeError):
    """Stop cleanly when another Gemini request would hit an exhausted quota."""

    def __init__(self, message: str = None):
        default_message = (
            "Gemini quota or request limit reached. Translation stopped to avoid "
            "wasting API calls or creating an incomplete PDF. Try again after the "
            "quota resets, or check the API project's limits."
        )
        super().__init__(message or default_message)


class GeminiDailyQuotaExceeded(GeminiQuotaExceeded):
    """Stop a translation cleanly when Gemini's daily free quota is exhausted."""
    def __init__(self, message: str = None):
        default_message = (
            "استُنفدت حصة Gemini اليومية المجانية. "
            "أوقفت المهمة لتجنب إنشاء ملف ناقص. "
            "أعد المحاولة بعد تجدد الحصة أو فعّل الفوترة في Google AI Studio."
        )
        super().__init__(message or default_message)

def translate_text(text: str, api_key: str, model_name: str = None) -> str:
    """Translate academic English text to Arabic with Gemini."""
    if not text or not text.strip():
        return ""
    model_name = model_name or _get_model()
    client = _get_client(api_key)
    prompt = f"""You are a professional academic translator specializing in scientific and forensic documents. Translate the text from English to formal Arabic.
Preserve its meaning, terminology, paragraph structure, list markers, and numbering. Return only the Arabic translation.

Text:
{text}
"""
    for attempt in range(4):
        try:
            response = _generate_content(
                client, model=model_name, contents=prompt,
            )
            translated = (response.text or "").strip()
            if translated:
                return translated
            raise ValueError("Gemini returned an empty translation")
        except Exception as exc:
            if _is_quota_error(exc) or _is_daily_quota_error(exc) or _is_timeout_error(exc):
                next_model = _switch_to_fallback(model_name)
                if next_model != model_name or attempt < 3:
                    logger.info("Switching to fallback model %s in translate_text after: %s", next_model, exc)
                    model_name = next_model
                    time.sleep(2)
                    continue
            if _is_daily_quota_error(exc):
                raise GeminiDailyQuotaExceeded() from exc
            if _is_request_limit_error(exc):
                raise GeminiQuotaExceeded() from exc
            if attempt < 3:
                logger.warning("Translation attempt failed: %s", exc)
                time.sleep(2)
            else:
                logger.error("Translation failed after retries: %s", exc)
                return f"Error translating (Network/API): {exc}"
    return ""


def generate_study_material(source_text: str, material_type: str, api_key: str,
                           question_count: int = 20, difficulty: str = "medium",
                           answer_key: str = "separate") -> str:
    """Generate English study material grounded in extracted lecture text."""
    if material_type not in {"quiz", "mind_map", "summary", "glossary"}:
        raise ValueError("Unsupported study material type")
    if not source_text or not source_text.strip():
        raise ValueError("No lecture text was provided")

    if material_type == "quiz":
        key_instruction = ("Put the answer key in a separate section after the exact marker '=== ANSWER KEY ==='."
                          if answer_key == "separate" else "Put the answer key at the end after '=== ANSWER KEY ==='.")
        instructions = f"""Create an English practice test based only on the lecture.
Create exactly {question_count} questions at {difficulty} difficulty, balanced across MCQ, True/False, and Fill in the Blanks.
Use clear section headings, number every question, and provide a concise answer key. {key_instruction}
For MCQs provide four options (A-D) and one correct answer. Include no answers beside the questions.
Do not invent facts absent from the lecture. Begin with '=== QUESTIONS ==='."""
    elif material_type == "mind_map":
        instructions = """Create a clear, detailed mind map in English based only on the lecture.
Use a central topic followed by nested branches and sub-branches, formatted as an indented outline.
Group related concepts, include key definitions and relationships, and keep each point concise.
Do not invent facts absent from the lecture. Use one topic per line and indent child branches by two spaces."""
    elif material_type == "summary":
        instructions = """Write a clear English study summary of this lecture.
Include a short overview, organized section summaries, key takeaways, and important terms.
Keep it comprehensive but concise, and use only facts stated in the lecture."""
    else:
        instructions = """Create an English-Arabic glossary of the lecture's important technical and academic terms.
Format as a table with columns: English term | Arabic equivalent | Short English definition.
Include only terms that occur in or are clearly defined by the lecture; preserve technical accuracy."""

    prompt = f"""{instructions}

Lecture text:
{source_text[:60000]}
"""
    model_name = _get_model()
    client = _get_client(api_key)
    for attempt in range(3):
        try:
            response = _generate_content(
                client, model=model_name, contents=prompt,
            )
            result = (response.text or "").strip()
            if result:
                return result
            raise ValueError("The model returned an empty response")
        except Exception as exc:
            if _is_quota_error(exc) or _is_timeout_error(exc):
                next_model = _switch_to_fallback(model_name)
                if next_model != model_name:
                    model_name = next_model
                    continue
            if attempt == 2:
                raise RuntimeError("Could not generate study material") from exc
            logger.warning("Study material generation attempt failed: %s", exc)
            time.sleep(2)
    raise RuntimeError("Could not generate study material")


def translate_single_item(item: Tuple[int, str, str]) -> Tuple[int, str, str]:
    """Helper for concurrent execution via ThreadPoolExecutor"""
    index, text, api_key = item
    if not text.strip():
        return index, text, ""
    try:
        translation = translate_text(text, api_key)
        if translation.startswith("Error translating") or translation.startswith("Error:"):
            logger.error("Translation unavailable for paragraph %s after model retries", index)
            translation = "[تعذرت الترجمة تلقائيًا؛ أعد المحاولة]"
        return index, text, translation
    except GeminiQuotaExceeded:
        raise
    except Exception as e:
        logger.error(f"Error in translate_single_item index {index}: {e}")
        return index, text, "[تعذرت الترجمة تلقائيًا؛ أعد المحاولة]"

def generate_document_summary(full_text: str, api_key: str, model_name: str = None) -> str:
    """Summarize a document's subject and terminology to guide consistent translation."""
    if not full_text or not full_text.strip():
        return ""
    model_name = model_name or _get_model()
    client = _get_client(api_key)
    prompt = """Summarize this academic document in no more than three sentences. Identify its field and key terminology. This summary will guide a later English-to-Arabic translation.

Document text:
""" + full_text[:15000]
    for attempt in range(3):
        try:
            response = _generate_content(
                client, model=model_name, contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as exc:
            if _is_daily_quota_error(exc):
                raise GeminiDailyQuotaExceeded() from exc
            # The summary is optional context. A 504 should not make the user
            # wait through several 45-second requests before paragraph
            # translation can begin.
            if _is_timeout_error(exc):
                logger.warning(
                    "Document summary timed out; continuing translation without summary context"
                )
                return ""
            if _is_quota_error(exc):
                next_model = _switch_to_fallback(model_name)
                if next_model != model_name:
                    model_name = next_model
                    if attempt == 0:
                        continue
            if _is_request_limit_error(exc):
                raise GeminiQuotaExceeded() from exc
            logger.warning("Document summary attempt %s failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(1)
    return ""


class TranslationItem(BaseModel):
    index: int = Field(description="The exact index from the input.")
    text: str = Field(description="The translated Arabic text.")

class TranslationBatchResponse(BaseModel):
    translations: List[TranslationItem] = Field(description="The list of translated items, maintaining the exact indices provided in the input.")

def translate_batch(batch: List[Tuple[int, str]], api_key: str, context_summary: str = "", model_name: str = None) -> List[Tuple[int, str, str]]:
    """Translate a batch of indexed paragraphs with Gemini and preserve order."""
    if not batch:
        return []
    model_name = model_name or _get_model()
    client = _get_client(api_key)
    valid = [{"index": idx, "text": text} for idx, text in batch if text and text.strip()]
    empty = [(idx, text, "") for idx, text in batch if not text or not text.strip()]
    if not valid:
        return empty
    context = f"Document context: {context_summary}\n" if context_summary else ""
    prompt = f"""Translate each English segment into formal academic Arabic.
{context}Maintain each segment separately, preserve numbering and list markers, and do not add explanations.
Return valid JSON exactly in this shape: {{"translations":[{{"index":0,"text":"Arabic translation"}}]}}.
Return exactly one item for each supplied index.

Input:
{json.dumps(valid, ensure_ascii=False)}
"""
    for attempt in range(4):
        try:
            response = _generate_content(
                client, model=model_name, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TranslationBatchResponse,
                    temperature=0.1,
                ),
            )
            data = json.loads(response.text or "{}")
            translated = {int(item["index"]): item["text"] for item in data.get("translations", [])}
            if any(
                item["index"] not in translated
                or not str(translated[item["index"]] or "").strip()
                for item in valid
            ):
                raise ValueError("Gemini omitted or returned an empty paragraph translation")
            results = empty + [(item["index"], item["text"], translated[item["index"]]) for item in valid]
            return sorted(results, key=lambda result: result[0])
        except Exception as exc:
            if _is_quota_error(exc) or _is_daily_quota_error(exc) or _is_timeout_error(exc):
                next_model = _switch_to_fallback(model_name)
                if next_model != model_name or attempt < 3:
                    logger.info("Switching to fallback model %s in translate_batch after: %s", next_model, exc)
                    model_name = next_model
                    time.sleep(2)
                    continue
            if _is_daily_quota_error(exc):
                raise GeminiDailyQuotaExceeded() from exc
            if _is_request_limit_error(exc):
                raise GeminiQuotaExceeded() from exc
            logger.warning("Gemini batch attempt %s failed: %s", attempt + 1, exc)
            if attempt < 3:
                time.sleep(2)
                continue
            if len(batch) > 1:
                midpoint = len(batch) // 2
                logger.warning(
                    "Gemini batch failed after retries; splitting %s paragraphs into smaller batches",
                    len(batch),
                )
                smaller_results = translate_batch(
                    batch[:midpoint], api_key, context_summary, model_name
                )
                smaller_results.extend(translate_batch(
                    batch[midpoint:], api_key, context_summary, model_name
                ))
                return sorted(smaller_results, key=lambda result: result[0])

            logger.error("Single-paragraph batch failed; falling back to individual translation")
            return [translate_single_item((batch[0][0], batch[0][1], api_key))]
    return []


def _build_translation_batches(paragraphs: List[str]) -> List[List[Tuple[int, str]]]:
    """Build fewer, balanced initial requests without exceeding the batch cap."""
    indexed = list(enumerate(paragraphs))
    if not indexed:
        return []
    usual_count = (len(indexed) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    batch_count = min(config.MAX_TRANSLATION_BATCHES, usual_count)
    batch_size = (len(indexed) + batch_count - 1) // batch_count
    return [indexed[i:i + batch_size] for i in range(0, len(indexed), batch_size)]


def get_translation_batch_count(paragraph_count) -> int:
    """Return the initial request count, accounting for repeated paragraphs when supplied."""
    if isinstance(paragraph_count, int):
        unique_count = max(0, paragraph_count)
    else:
        unique_count = len({
            _normalize_translation_text(paragraph)
            for paragraph in paragraph_count
            if paragraph and paragraph.strip()
        })
    return len(_build_translation_batches([""] * unique_count))


def translate_paragraphs_concurrently(paragraphs: List[str], api_key: str, context_summary: str = "", max_workers: int = config.MAX_TRANSLATION_WORKERS) -> Generator[Tuple[int, str, str], None, None]:
    """Translate unique paragraphs once, reuse local cache hits, and restore original positions."""
    # Stable cache key model: fallback switches must not hide saved translations
    # from the next bot run.
    model_name = config.MODEL_NAME
    occurrences = {}
    unique_keys = []
    unique_texts = []

    for index, text in enumerate(paragraphs):
        if not text or not text.strip():
            yield index, text, ""
            continue
        cache_key = _translation_cache_key(text, context_summary, model_name)
        if cache_key not in occurrences:
            occurrences[cache_key] = []
            unique_keys.append(cache_key)
            unique_texts.append(text)
        occurrences[cache_key].append((index, text))

    cached = _load_cached_translations(unique_keys)
    pending_keys = []
    pending_texts = []
    for cache_key, text in zip(unique_keys, unique_texts):
        translation = cached.get(cache_key)
        if translation and translation.strip():
            for original_index, original_text in occurrences[cache_key]:
                yield original_index, original_text, translation
        else:
            pending_keys.append(cache_key)
            pending_texts.append(text)

    pending_index_to_key = {index: key for index, key in enumerate(pending_keys)}
    batches = _build_translation_batches(pending_texts)
    completed_keys = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(translate_batch, batch, api_key, context_summary): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                results = future.result()
                cache_entries = []
                for unique_index, _original, translation in results:
                    cache_key = pending_index_to_key[unique_index]
                    cache_entries.append((cache_key, translation))
                    completed_keys.add(cache_key)
                _save_cached_translations(cache_entries)
                for unique_index, _original, translation in results:
                    cache_key = pending_index_to_key[unique_index]
                    for original_index, original_text in occurrences[cache_key]:
                        yield original_index, original_text, translation
            except GeminiQuotaExceeded:
                for pending in future_to_batch:
                    pending.cancel()
                raise
            except Exception as exc:
                logger.error("Batch execution exception: %s", exc)
                for unique_index, text in future_to_batch[future]:
                    cache_key = pending_index_to_key[unique_index]
                    try:
                        _result_index, _original, translation = translate_single_item(
                            (unique_index, text, api_key)
                        )
                        if _is_cacheable_translation(translation):
                            _save_cached_translations([(cache_key, translation)])
                        completed_keys.add(cache_key)
                        for original_index, original_text in occurrences[cache_key]:
                            yield original_index, original_text, translation
                    except GeminiQuotaExceeded:
                        for pending in future_to_batch:
                            pending.cancel()
                        raise
                    except Exception as item_error:
                        logger.error("Paragraph %s recovery failed: %s", unique_index, item_error)
                        completed_keys.add(cache_key)
                        for original_index, original_text in occurrences[cache_key]:
                            yield original_index, original_text, "[Translation unavailable; please retry]"

    # Recover any omitted item from an otherwise successful batch response.
    for unique_index, cache_key in enumerate(pending_keys):
        if cache_key in completed_keys:
            continue
        text = pending_texts[unique_index]
        logger.warning("Translation result missing for unique paragraph %s; retrying individually", unique_index)
        try:
            _result_index, _original, translation = translate_single_item(
                (unique_index, text, api_key)
            )
            if _is_cacheable_translation(translation):
                _save_cached_translations([(cache_key, translation)])
        except GeminiQuotaExceeded:
            raise
        except Exception as item_error:
            logger.error("Paragraph %s recovery failed: %s", unique_index, item_error)
            translation = "[Translation unavailable; please retry]"
        for original_index, original_text in occurrences[cache_key]:
            yield original_index, original_text, translation


class TableTranslationResponse(BaseModel):
    translated_rows: List[List[str]] = Field(description="The 2D array of translated arabic strings, corresponding exactly to the input table cells.")

def translate_table(table_rows: List[List[str]], api_key: str, context_summary: str = "", model_name: str = None) -> List[List[str]]:
    """Translate a table cohesively while preserving its dimensions."""
    if not table_rows:
        return []
    model_name = model_name or _get_model()
    table_cache_key = _translation_cache_key(
        json.dumps(table_rows, ensure_ascii=False),
        "table-v1:" + context_summary,
        config.MODEL_NAME,
    )
    cached_table = _load_cached_translations([table_cache_key]).get(table_cache_key)
    if cached_table:
        try:
            rows = json.loads(cached_table)
            if len(rows) == len(table_rows) and all(
                len(translated_row) == len(original_row)
                for translated_row, original_row in zip(rows, table_rows)
            ):
                return rows
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring invalid cached table translation")
    client = _get_client(api_key)
    context = f"Document context: {context_summary}\n" if context_summary else ""
    prompt = f"""Translate this table from English to formal Arabic.
{context}Preserve the exact row and column count, empty cells, HTML tags, and consistent terminology. Return JSON exactly as {{"translated_rows":[["..."]]}}.

Table:
{json.dumps(table_rows, ensure_ascii=False)}
"""
    for attempt in range(4):
        try:
            response = _generate_content(
                client, model=model_name, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TableTranslationResponse,
                    temperature=0.1,
                ),
            )
            rows = json.loads(response.text or "{}").get("translated_rows", [])
            if len(rows) != len(table_rows) or any(len(a) != len(b) for a, b in zip(rows, table_rows)):
                raise ValueError("Gemini returned a table with mismatched dimensions")
            _save_cached_translations([(table_cache_key, json.dumps(rows, ensure_ascii=False))])
            return rows
        except Exception as exc:
            if _is_quota_error(exc) or _is_daily_quota_error(exc) or _is_timeout_error(exc):
                next_model = _switch_to_fallback(model_name)
                if next_model != model_name or attempt < 3:
                    logger.info("Switching to fallback model %s in translate_table after: %s", next_model, exc)
                    model_name = next_model
                    time.sleep(2)
                    continue
            if _is_daily_quota_error(exc):
                raise GeminiDailyQuotaExceeded() from exc
            if _is_request_limit_error(exc):
                raise GeminiQuotaExceeded() from exc
            logger.warning("Table translation attempt %s failed: %s", attempt + 1, exc)
            if attempt < 3:
                time.sleep(2)
    logger.error("Table translation failed; falling back to cell-by-cell Gemini translation")
    return [[translate_text(str(cell), api_key) if str(cell or "").strip() else "" for cell in row] for row in table_rows]


class EnglishTextExtractionResponse(BaseModel):
    paragraphs: List[str] = Field(description="List of extracted english paragraphs.")


class OCRTableRow(BaseModel):
    cells_english: List[str] = Field(description="English text for each cell in the row.")
    cells_arabic:  List[str] = Field(description="Arabic translation for each cell in the same row.")

class OCRTranslationPair(BaseModel):
    type: str = Field(description="Content type: 'text', 'table', or 'image'.")
    # --- text fields ---
    english: str = Field(default="", description="For type='text': extracted English paragraph with HTML tags.")
    arabic:  str = Field(default="", description="For type='text': Arabic translation using <br/> between list items.")
    # --- table fields ---
    table_rows: List[OCRTableRow] = Field(default=[], description="For type='table': list of rows with bilingual cells.")
    # --- image bounding box fields (for type='image') ---
    x_pct: float = Field(default=0, description="For type='image': left edge as percent of page width (0-100).")
    y_pct: float = Field(default=0, description="For type='image': top edge as percent of page height (0-100).")
    w_pct: float = Field(default=0, description="For type='image': width as percent of page width (0-100).")
    h_pct: float = Field(default=0, description="For type='image': height as percent of page height (0-100).")

class OCRTranslationResponse(BaseModel):
    items: List[OCRTranslationPair] = Field(description="List of content items found on the page.")


def extract_and_translate_page(image_bytes: bytes, api_key: str) -> List[dict]:
    """
    Extracts English text AND translates to Arabic in a SINGLE API call.
    Supports three content types: text paragraphs, tables, and images.
    Returns a list of dicts matching OCRTranslationPair structure.
    """
    model_name = _get_model()
    image_digest = hashlib.sha256(image_bytes).hexdigest()
    page_cache_key = _translation_cache_key(
        image_digest, "ocr-page-v1", config.MODEL_NAME
    )
    cached_page = _load_cached_translations([page_cache_key]).get(page_cache_key)
    if cached_page:
        try:
            cached_items = json.loads(cached_page)
            if isinstance(cached_items, list):
                return cached_items
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring invalid cached OCR page result")
    client = _get_client(api_key)

    prompt = """Analyze this scanned document image carefully. Identify ALL content blocks in reading order.
    Extract the English content and translate it into formal academic Arabic in the same response.

    For each block, determine its type and return it accordingly:

    === TYPE 1: TEXT PARAGRAPH ===
    Return: {"type": "text", "english": "...", "arabic": "..."}
    - Extract English text with HTML formatting for bold, italic, and colored spans.
    - Preserve bullet and numbered-list structure with <br/> separators.
    - Translate all extracted English into Arabic. Preserve list line breaks with <br/> and do not repeat English.

    === CRITICAL: BLOCK SEPARATION RULES ===
    - Keep headings, paragraphs, and bullet items as separate blocks.
    - Use visual font/style changes to identify separate blocks.

    === TYPE 2: TABLE ===
    Return: {"type": "table", "table_rows": [{"cells_english": [...], "cells_arabic": [...]} , ...]}
    - Extract each cell in English and preserve the table dimensions.
    - Translate every cell, preserving the exact number of rows and columns.

    === TYPE 3: IMAGE / DIAGRAM / FIGURE ===
    Return: {"type": "image", "x_pct": ..., "y_pct": ..., "w_pct": ..., "h_pct": ...}
    - Use this for: photographs, charts, diagrams, decorative graphics, logos, figures.
    - ANY block that is primarily visual with no meaningful extractable text.
    - Do NOT attempt to transcribe text from within graphs or chart axes.
    - You MUST provide the bounding box of the image as percentages of the page dimensions:
      • x_pct: left edge as percent of page width (0-100)
      • y_pct: top edge as percent of page height (0-100)
      • w_pct: width as percent of page width (must be > 0)
      • h_pct: height as percent of page height (must be > 0)
    - The bounding box must tightly enclose the image/diagram area.

    RULES:
    - Process blocks in top-to-bottom, left-to-right reading order.
    - Ignore Arabic text already present in the image.
    - Do NOT add markdown, backticks, or commentary.
    - Return ALL blocks including images WITH their bounding boxes.
    - NEVER merge a heading/title with the paragraph that follows it. Keep them as separate items.
    - Strip bullet symbols (■, □, •, ●) from the "english" field text. They are formatting, not content.
    """

    retries = 3
    for attempt in range(retries):
        try:
            response = _generate_content(
                client,
                model=model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=OCRTranslationResponse,
                    temperature=0.1
                )
            )

            if not response.text:
                return []

            response_data = json.loads(response.text)
            items = response_data.get("items", [])
            if items:
                _save_cached_translations([
                    (page_cache_key, json.dumps(items, ensure_ascii=False))
                ])
            return items
        except Exception as e:
            if _is_quota_error(e):
                new_model = _switch_to_fallback(model_name)
                if new_model != model_name:
                    model_name = new_model
                    continue
            logger.warning(f"OCR+Translation attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"OCR+Translation failed after {retries} attempts: {e}")
                return []
    return []


# ====== Slide Translation Models & Function ======

class SlideTextBlock(BaseModel):
    block_id: int = Field(description="Sequential block order on the slide (0-indexed, top to bottom).")
    text_en: str = Field(description="The English text of this block, preserving bullet points and numbering.")
    text_ar: str = Field(description="Accurate Arabic translation of text_en.")
    is_title: bool = Field(description="True if this block is a title/heading (typically bold and/or colored differently).")
    font_color_hex: str = Field(default="#000000", description="The dominant foreground color of this text block as a hex string, e.g. '#cc0000' for red titles.")

class SlideTranslationResponse(BaseModel):
    blocks: List[SlideTextBlock] = Field(description="All text blocks on the slide in reading order.")

def translate_slide_image(image_bytes: bytes, api_key: str) -> List[dict]:
    """Legacy unpositioned slide translation (kept for compatibility)."""
    result = translate_slide_positioned(image_bytes, api_key)
    return result.get('blocks', [])


# ====== Slide Mirror Translation (Positioned) ======

class SlideBlockPositioned(BaseModel):
    block_id: int = Field(description="Sequential block index (0-based, top-to-bottom left-to-right).")
    text_en: str = Field(description="Exact English text of this block. Preserve line breaks as \n.")
    text_ar: str = Field(description="Professional Arabic translation. Preserve list structure.")
    x_pct: float = Field(description="Left edge of text block as percent of slide width (0-100).")
    y_pct: float = Field(description="Top edge of text block as percent of slide height (0-100).")
    w_pct: float = Field(description="Width of text block as percent of slide width (1-100).")
    h_pct: float = Field(description="Height of text block as percent of slide height (1-100).")
    font_color_hex: str = Field(default="#000000", description="Text color hex, e.g. '#cc2222' for red, '#000000' for black.")
    bg_color_hex: str = Field(default="#ffffff", description="Background color directly behind this text block, e.g. '#ebf5fb'.")
    is_bold: bool = Field(default=False, description="True if the text appears bold.")
    font_size_pt: int = Field(default=12, description="Estimated original font size in points (e.g. 10, 12, 14, 18, 24, 28, 36).")
    text_align: str = Field(default="left", description="Text alignment: 'left', 'right', or 'center'.")

class SlidePositionedResponse(BaseModel):
    slide_bg_color: str = Field(default="#ffffff", description="Main background color of the entire slide (hex).")
    blocks: List[SlideBlockPositioned] = Field(description="All text blocks with positions and translations.")

def translate_slide_positioned(image_bytes: bytes, api_key: str) -> dict:
    """
    Analyzes a slide image with AI Vision and returns positioned bilingual text blocks.
    Each block includes: bounding box (%), text, translation, font color, bg color, size, alignment.
    Returns a dict: {'slide_bg_color': '#...', 'blocks': [...]}, or {} on failure.
    """
    model_name = _get_model()
    client = _get_client(api_key)

    prompt = """Analyze this presentation slide image with precision.

Your goal: Extract every text block with its exact visual position and translate it into Arabic.

For EACH text block return:
- block_id: 0-based sequential index (top to bottom, left to right)
- text_en: Exact text content (use \n for line breaks within a block)
  • If the slide has bilingual content (Arabic + English), extract only the English side
  • If a block is purely in Arabic already, still put it in text_en
- text_ar: Accurate, concise Arabic translation. Preserve list markers and numbering.
- x_pct: left edge of text bounding box as % of image width (0–100)
- y_pct: top edge of text bounding box as % of image height (0–100)
- w_pct: width of text bounding box as % of image width (must be > 0)
- h_pct: height of text bounding box as % of image height (must be > 0)
- font_color_hex: exact CSS hex color of the text (e.g. "#cc1111" for red titles, "#000000" for black body text)
- bg_color_hex: the background color directly behind this text (e.g. "#ebf5fb", "#ffffff", "#003366")
- is_bold: true if font appears bold/heavy
- font_size_pt: estimated font size in points (common: 10, 12, 14, 18, 24, 28, 36)
- text_align: "left", "right", or "center"

Also return:
- slide_bg_color: the main background hex color of the entire slide

RULES:
- Return ALL visible text blocks, including small labels, page numbers, footers
- Each separate text box = one block (do NOT merge across boxes)
- Bounding boxes must tightly enclose the text (not the whole slide)
- DO NOT include decorative elements with no text
"""

    retries = 3
    for attempt in range(retries):
        try:
            response = _generate_content(
                client,
                model=model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SlidePositionedResponse,
                    temperature=0.1
                )
            )

            if not response.text:
                return {}

            return json.loads(response.text)  # Gemini returns translations with positions.

        except Exception as e:
            if _is_quota_error(e):
                new_model = _switch_to_fallback(model_name)
                if new_model != model_name:
                    model_name = new_model
                    continue
            logger.warning(f"Positioned slide translation attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Positioned slide translation failed after {retries} attempts: {e}")
                return {}
    return {}


# ====== Slide Extraction Only (positions, no translation) ======

class SlideBlockExtracted(BaseModel):
    block_id: int = Field(description="Sequential block index (0-based, top-to-bottom left-to-right).")
    text_en: str = Field(description="Exact English text of this block. Preserve line breaks as \\n.")
    x_pct: float = Field(description="Left edge of text block as percent of slide width (0-100).")
    y_pct: float = Field(description="Top edge of text block as percent of slide height (0-100).")
    w_pct: float = Field(description="Width of text block as percent of slide width (1-100).")
    h_pct: float = Field(description="Height of text block as percent of slide height (1-100).")
    font_color_hex: str = Field(default="#000000", description="Text color hex.")
    bg_color_hex: str = Field(default="#ffffff", description="Background color directly behind this text block.")
    is_bold: bool = Field(default=False, description="True if the text appears bold.")
    font_size_pt: int = Field(default=12, description="Estimated original font size in points.")
    text_align: str = Field(default="left", description="Text alignment: 'left', 'right', or 'center'.")

class SlideExtractionResponse(BaseModel):
    slide_bg_color: str = Field(default="#ffffff", description="Main background color of the entire slide (hex).")
    blocks: List[SlideBlockExtracted] = Field(description="All text blocks with positions.")

def extract_slide_blocks(image_bytes: bytes, api_key: str) -> dict:
    """
    Extracts positioned text blocks from a slide image using Gemini Vision.
    Extracts text and coordinates; translation is performed separately with Gemini.
    Returns {'slide_bg_color': '#...', 'blocks': [{block fields}]}, or {} on failure.
    """
    model_name = _get_model()
    client = _get_client(api_key)

    prompt = """Analyze this presentation slide image. Extract every visible text block with its exact position.

For EACH text block return:
- block_id: 0-based sequential index (top to bottom, left to right)
- text_en: Exact text content (use \\n for line breaks within a block)
  • If the slide has bilingual content, extract only the English side
  • If a block is purely Arabic, still put it in text_en
- x_pct: left edge as % of image width (0–100)
- y_pct: top edge as % of image height (0–100)
- w_pct: width as % of image width (must be > 0)
- h_pct: height as % of image height (must be > 0)
- font_color_hex: exact CSS hex color of the text (e.g. "#cc1111" for red, "#000000" for black)
- bg_color_hex: background color directly behind this text
- is_bold: true if font appears bold/heavy
- font_size_pt: estimated font size in points (common: 10, 12, 14, 18, 24, 28, 36)
- text_align: "left", "right", or "center"

Also return:
- slide_bg_color: the main background hex color of the entire slide

RULES:
- Return ALL visible text blocks including small labels, page numbers, footers
- Each separate text box = one block (do NOT merge across boxes)
- Bounding boxes must tightly enclose the text
- DO NOT include decorative elements with no text
"""

    retries = 3
    for attempt in range(retries):
        try:
            response = _generate_content(
                client,
                model=model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            types.Part.from_text(text=prompt)
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SlideExtractionResponse,
                    temperature=0.1
                )
            )

            if not response.text:
                return {}

            return json.loads(response.text)

        except Exception as e:
            err_str = str(e).lower()
            if _is_quota_error(e):
                new_model = _switch_to_fallback(model_name)
                if new_model != model_name:
                    model_name = new_model
                    continue
            # Server disconnect = payload too large or transient network error; back off longer
            is_disconnect = any(kw in err_str for kw in [
                'server disconnected', 'connection reset', 'broken pipe',
                'remote end closed', 'eof occurred', 'connection aborted'
            ])
            sleep_time = (4 if is_disconnect else 2) ** (attempt + 1)
            logger.warning(f"Slide extraction attempt {attempt+1} failed ({type(e).__name__}): {e}. Retrying in {sleep_time}s...")
            if attempt < retries - 1:
                time.sleep(sleep_time)
            else:
                logger.error(f"Slide extraction failed after {retries} attempts: {e}")
                return {}
    return {}


def translate_texts_gemini(texts: List[str], api_key: str) -> List[str]:
    """Translate slide text through Gemini while preserving the input order."""
    if not texts:
        return []
    results = [""] * len(texts)
    for idx, original, translated in translate_paragraphs_concurrently(texts, api_key):
        results[idx] = translated
    return results


def extract_english_text_from_image(image_bytes: bytes, api_key: str) -> List[str]:
    """
    Extracts ONLY English text paragraphs from a scanned document image using Gemini Vision.
    With automatic model fallback on quota exhaustion.
    """
    model_name = _get_model()
    
    try:
        client = _get_client(api_key)
        
        prompt = """Analyze this scanned document image. 
        Extract ONLY the text written in the English language. Ignore any text in Arabic or other languages.
        
        CRITICAL STYLING INSTRUCTIONS:
        - If a word or phrase is bold in the image, wrap it in <b>...</b>.
        - If a word or phrase is italic in the image, wrap it in <i>...</i>.
        - If a word or phrase has a distinct color (NOT black), wrap it in <font color='#RRGGBB'>...</font> using the closest hex color you can identify from the image. Do NOT wrap black text in font tags.
        - Exactly preserve any bullet points (•, -, 1., a.), line breaks, and list structures.
        - Do not add any backticks, markdown code blocks, or extra text.
        
        Group the extracted English text into logical paragraphs or list blocks as they appear in the source.
        Do not translate the text here. Just transcribe the English portions with the HTML tags.
        Return the result as a JSON array of strings, where each string is a fully formatted paragraph.
        """
        
        retries = 2
        for attempt in range(retries):
            try:
                response = _generate_content(
                    client,
                    model=model_name,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                                types.Part.from_text(text=prompt)
                            ]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EnglishTextExtractionResponse,
                        temperature=0.1
                    )
                )
                
                if not response.text:
                    return []
                    
                response_data = json.loads(response.text)
                paragraphs = response_data.get("paragraphs", [])
                return paragraphs
                
            except Exception as e:
                if _is_quota_error(e):
                    new_model = _switch_to_fallback(model_name)
                    if new_model != model_name:
                        model_name = new_model
                        continue  # Retry with next model
                raise  # Re-raise for outer handler
        
        return []
        
    except Exception as e:
        logger.error(f"English text extraction from scanned image failed: {e}")
        return []
