import fitz  # PyMuPDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.units import inch
import arabic_reshaper
from bidi.algorithm import get_display
import os
import urllib.request
import ssl
import logging
import threading
import re
import tempfile
import config
from translator import GeminiQuotaExceeded, translate_text, translate_paragraphs_concurrently, get_translation_batch_count, extract_english_text_from_image, extract_and_translate_page, translate_slide_image, translate_table
from pptx import Presentation
import io
from PIL import Image as PILImage
from reportlab.platypus import Image as RLImage
from reportlab.lib.utils import ImageReader

# Setup logging
logger = logging.getLogger(__name__)
_FLOW_FONT_LOCK = threading.Lock()

def download_arabic_font():
    """Downloads the Arabic font if it doesn't exist."""
    success = True
    
    # Download Regular
    if not os.path.exists(config.ARABIC_FONT_FILE):
        logger.info(f"Downloading {config.ARABIC_FONT_FILE}...")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(config.ARABIC_FONT_URL, context=ctx, timeout=30) as response, open(config.ARABIC_FONT_FILE, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            logger.info("Font downloaded successfully.")
        except Exception as e:
            logger.error(f"Error downloading font: {e}")
            success = False
            
    return success

def register_fonts():
    """Registers the Arabic font for ReportLab."""
    if download_arabic_font():
        try:
            pdfmetrics.registerFont(TTFont(config.ARABIC_FONT_NAME, config.ARABIC_FONT_FILE))
            from reportlab.pdfbase.pdfmetrics import registerFontFamily
            registerFontFamily(config.ARABIC_FONT_NAME, normal=config.ARABIC_FONT_NAME, bold=config.ARABIC_FONT_NAME, italic=config.ARABIC_FONT_NAME, boldItalic=config.ARABIC_FONT_NAME)
            return True
        except Exception as e:
            logger.error(f"Error registering font: {e}")
            return False
    else:
        logger.warning("Font file not found. Arabic support may be limited.")
        return False

def is_scanned_pdf(pdf_path_or_stream) -> bool:
    """Auto-detects whether a PDF is scanned (image-based) or has selectable text.
    Checks up to the first 5 pages. If the majority have less than 50 chars
    of extractable text, the PDF is considered scanned."""
    try:
        if isinstance(pdf_path_or_stream, str):
            doc = fitz.open(pdf_path_or_stream)
        else:
            if hasattr(pdf_path_or_stream, 'seek'):
                pdf_path_or_stream.seek(0)
            doc = fitz.open(stream=pdf_path_or_stream.read(), filetype="pdf")
            pdf_path_or_stream.seek(0)  # Reset for later use
    except Exception as e:
        logger.warning(f"is_scanned_pdf: cannot open file: {e}")
        return False

    sample_count = min(5, len(doc))
    text_pages, image_pages = 0, 0
    for i in range(sample_count):
        text = doc[i].get_text().strip()
        if len(text) < 50:
            image_pages += 1
        else:
            text_pages += 1
    doc.close()

    is_scanned = image_pages > text_pages
    logger.info(f"Auto-detect: {sample_count} pages sampled — {text_pages} text, {image_pages} image → {'SCANNED' if is_scanned else 'TEXT'}")
    return is_scanned


def _adaptive_dpi(page) -> int:
    """Returns a higher DPI for small pages (e.g. slides) to improve OCR accuracy."""
    w, h = page.rect.width, page.rect.height
    area = w * h
    if area < 300_000:     # Very small page
        return 250
    elif area < 500_000:   # Medium-small page
        return 200
    return 150             # Standard


def _compress_for_api(img_bytes: bytes, max_size_kb: int = 500) -> bytes:
    """Compresses a JPEG image to stay under max_size_kb. Reduces quality iteratively."""
    if len(img_bytes) / 1024 <= max_size_kb:
        return img_bytes
    img = PILImage.open(io.BytesIO(img_bytes))
    quality = 80
    while quality >= 30:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        if buf.tell() / 1024 <= max_size_kb:
            return buf.getvalue()
        quality -= 10
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=30)
    return buf.getvalue()


def _validate_ocr_result(items: list, page_num: int) -> bool:
    """Checks whether an OCR result looks reasonable. Returns False if likely failed."""
    if not items:
        return True  # Empty page is acceptable
    text_items = [it for it in items if isinstance(it, dict) and it.get('type') == 'text']
    if not text_items:
        return True  # Only images/tables is acceptable
    total_en = sum(len(it.get('english', '')) for it in text_items)
    has_arabic = any(it.get('arabic', '').strip() for it in text_items)
    if total_en < 10:
        logger.warning(f"Page {page_num+1}: OCR result suspiciously short ({total_en} chars)")
        return False
    if not has_arabic:
        logger.warning(f"Page {page_num+1}: OCR returned English but no Arabic translation")
        return False
    return True


def extract_text_from_pdf(pdf_file, api_key=None, extract_images=True, is_scanned=False, status_text=None, progress_bar=None):
    """
    Extracts content (Text + Images) from a PDF file using PyMuPDF.
    Returns a list of dictionaries:
    - {'type': 'text', 'content': '<b>Hello</b>', 'plain': 'Hello'}
    - {'type': 'image', 'data': bytes, 'width': w, 'height': h}
    """
    try:
        if isinstance(pdf_file, str):
            doc = fitz.open(pdf_file)
        else:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except Exception as e:
        logger.error(f"Failed to open PDF stream: {e}")
        return []

    all_content = []
    
    # --- [NEW OCR MODE] ---
    if is_scanned and api_key:
        logger.info("OCR Mode enabled. Processing pages as images.")
        total_pages = len(doc)
        
        from PIL import ImageEnhance, ImageStat
        import concurrent.futures
        
        # --- Helper: Preprocess a single page into enhanced image bytes ---
        from PIL import ImageFilter
        
        def _preprocess_page(page, dpi=None):
            """Renders and preprocesses a page with adaptive DPI and advanced enhancement.
            Returns (color_bytes, is_blank)."""
            if dpi is None:
                dpi = _adaptive_dpi(page)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("jpeg")
            
            img = PILImage.open(io.BytesIO(img_bytes))
            
            # [OPT 3] Blank page detection — use grayscale stats only for detection
            grayscale = img.convert('L')
            stat = ImageStat.Stat(grayscale)
            mean_brightness = stat.mean[0]  # 0-255, white=255
            stddev = stat.stddev[0]
            if mean_brightness > 248 and stddev < 5:
                return None, True  # Blank page
            
            # [ENHANCED] Advanced preprocessing for scanned documents
            # 1. Light noise reduction
            img = img.filter(ImageFilter.MedianFilter(3))
            
            # 2. Sharpen text edges
            sharpener = ImageEnhance.Sharpness(img)
            img = sharpener.enhance(1.4)
            
            # 3. Boost contrast
            enhancer = ImageEnhance.Contrast(img)
            enhanced = enhancer.enhance(1.3)
            
            buf = io.BytesIO()
            enhanced.save(buf, format='JPEG', quality=85)
            
            # 4. Compress if too large for API efficiency
            final_bytes = _compress_for_api(buf.getvalue(), max_size_kb=500)
            return final_bytes, False
        
        # --- Helper: Process a single page (preprocess + combined OCR+Translation API call) ---
        def _ocr_single_page(args):
            """Preprocesses, OCRs and translates a single page in ONE API call.
            Returns (page_num, items) where items = [{'english', 'arabic'}].
            Includes quality validation with one retry on failure."""
            page_num, page, api_key_inner = args
            enhanced_bytes, is_blank = _preprocess_page(page)  # adaptive DPI
            if is_blank:
                logger.info(f"Page {page_num + 1}: Skipped (blank page)")
                return page_num, []
            # Single API call that does both OCR and Arabic translation
            items = extract_and_translate_page(enhanced_bytes, api_key_inner)
            
            # Validate quality — retry once with higher DPI if result is poor
            if not _validate_ocr_result(items, page_num):
                logger.info(f"Page {page_num + 1}: Retrying with higher DPI...")
                enhanced_bytes_hq, _ = _preprocess_page(page, dpi=250)
                if enhanced_bytes_hq:
                    items = extract_and_translate_page(enhanced_bytes_hq, api_key_inner)
            
            return page_num, items
        
        # [OPT 1] Concurrent OCR+Translation — process multiple pages simultaneously
        if status_text:
            status_text.info(f"Step 1/2: Reading & Translating {total_pages} scanned pages with AI Vision (concurrent)...")
        if progress_bar:
            progress_bar.progress(5)
        
        max_ocr_workers = min(config.MAX_OCR_WORKERS, total_pages)  # Use config, cap at page count
        page_args = [(i, doc[i], api_key) for i in range(total_pages)]
        
        results_by_page = {}
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_ocr_workers) as executor:
            PAGE_TIMEOUT_SEC = 120  # Max seconds to wait per page before giving up
            future_map = {executor.submit(_ocr_single_page, args): args[0] for args in page_args}
            
            for future in concurrent.futures.as_completed(future_map, timeout=PAGE_TIMEOUT_SEC * total_pages):
                page_num = future_map[future]
                try:
                    pn, paragraphs = future.result(timeout=PAGE_TIMEOUT_SEC)
                    results_by_page[pn] = paragraphs
                except concurrent.futures.TimeoutError:
                    logger.warning(f"OCR timed out for page {page_num + 1} after {PAGE_TIMEOUT_SEC}s — skipping.")
                    results_by_page[page_num] = []
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num + 1}: {e}")
                    results_by_page[page_num] = []
                
                completed += 1
                if status_text and progress_bar:
                    status_text.info(f"Step 1/2: AI Vision OCR & Translation... ({completed}/{total_pages} pages done)")
                    progress_bar.progress(int((completed / total_pages) * 66))
        
        # Reassemble in page order — items now have types: text / table / image
        raw_items = []
        for page_num in range(total_pages):
            for pair in results_by_page.get(page_num, []):
                if not isinstance(pair, dict):
                    continue

                item_type = pair.get('type', 'text')

                # --- type='image': skip entirely, not added to output ---
                if item_type == 'image':
                    logger.debug(f"Page {page_num + 1}: Skipping image/diagram block.")
                    continue

                # --- type='table': build a structured table item ---
                elif item_type == 'table':
                    table_rows_data = pair.get('table_rows', [])
                    if not table_rows_data:
                        continue

                    # Build parallel 2D arrays: rows (English) + translated_rows (Arabic)
                    rows_en = []
                    rows_ar = []
                    for row_obj in table_rows_data:
                        if isinstance(row_obj, dict):
                            cells_en = row_obj.get('cells_english', [])
                            cells_ar = row_obj.get('cells_arabic', [])
                        else:
                            # Pydantic object returned as-is
                            cells_en = getattr(row_obj, 'cells_english', [])
                            cells_ar = getattr(row_obj, 'cells_arabic', [])
                        rows_en.append([str(c) for c in cells_en])
                        rows_ar.append([str(c) for c in cells_ar])

                    if rows_en:
                        raw_items.append({
                            'type': 'table',
                            'rows': rows_en,
                            'translated_rows': rows_ar,  # Already translated
                            'page_num': page_num
                        })

                # --- type='text' (default): plain text paragraph ---
                else:
                    english_text = pair.get('english', '')
                    arabic_text  = pair.get('arabic', '')
                    clean_content = " ".join(english_text.splitlines()).strip()
                    if clean_content:
                        plain_text = re.sub(r'<[^>]+>', '', clean_content).strip()
                        raw_items.append({
                            'type': 'text',
                            'content': clean_content,
                            'plain': plain_text,
                            'translation': arabic_text,  # Already translated
                            'page_num': page_num
                        })

        
        # --- Repetitive Text Filtering (Headers, Footers, Page Numbers) ---
        # Only applicable to text items; tables pass through untouched.
        text_frequency = {}
        for item in raw_items:
            if item['type'] != 'text':
                continue
            t = item['plain'].strip()
            # Safelist common subheadings to avoid stripping them in OCR mode
            if t.lower() in {"location:", "location", "function:", "function", "example:", "example"}:
                continue
            if t and len(t) < 200:
                text_frequency[t] = text_frequency.get(t, 0) + 1
        
        repetition_threshold = max(2, int(total_pages * 0.25)) if total_pages > 3 else 999
        always_remove = {t for t, c in text_frequency.items() if c >= max(2, int(total_pages * 0.5))}
        repetitive_candidates = {t for t, c in text_frequency.items() if c >= repetition_threshold}
        
        page_num_patterns = [
            re.compile(r'^\d+$'),
            re.compile(r'^(page|pg\.?)\s*\|?\s*\d+$', re.IGNORECASE),
            re.compile(r'^\d+\s*[\/\-]\s*\d+$'),
            re.compile(r'^[\-\•]\s*\d+\s*[\-\•]$'),
            re.compile(r'^\d+\s*of\s*\d+$', re.IGNORECASE)
        ]
        
        seen_repetitive = set()
        for item in raw_items:
            # Tables always pass through — they have no 'plain' header/footer issue
            if item['type'] == 'table':
                all_content.append(item)
                continue

            plain = item['plain'].strip()
            
            if len(plain) < 20:
                if any(p.match(plain) for p in page_num_patterns):
                    continue
            
            if plain in always_remove:
                continue
            
            if plain in repetitive_candidates:
                if plain in seen_repetitive:
                    continue
                seen_repetitive.add(plain)
            
            all_content.append(item)
                    
        return all_content
    # --- [END OCR MODE] ---

    # 1. Frequency Analysis for Header/Footer Removal
    # We'll do a quick pass to find repetitive text blocks
    block_frequency = {}
    
    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] == 0: # Text block
                block_plain_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_plain_text += span["text"] + " "
                
                clean_text = " ".join(block_plain_text.splitlines()).strip()
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                # Only track frequencies for blocks in the top 25% or bottom 25% (Margins)
                y_center = (block["bbox"][1] + block["bbox"][3]) / 2.0
                page_height = page.rect.height
                is_margin = (page_height > 0) and (y_center / page_height < 0.25 or y_center / page_height > 0.75)
                
                if clean_text and len(clean_text) < 200 and is_margin:
                    block_frequency[clean_text] = block_frequency.get(clean_text, 0) + 1

    total_pages = len(doc)
    repetition_threshold = max(2, int(total_pages * 0.25)) if total_pages > 3 else 999 
    # Identify repetitive texts (candidates for headers/footers)
    # Lowered threshold to 25% of pages to catch more common headers
    repetitive_candidate_texts = {text for text, count in block_frequency.items() if count >= repetition_threshold}
    
    # Set to track what we've seen to allow the FIRST occurrence of a header (Title) but remove others
    # Update: If it's very repetitive (>= 50% pages), remove ALL occurrences including the first
    always_remove_texts = {text for text, count in block_frequency.items() if count >= max(2, int(total_pages * 0.5))}
    seen_repetitive_texts = set()
    
    # Page Number Regex Patterns
    # Matches: "1", "12", "Page 1", "1 / 4", "1-4", "- 1 -", "Page | 1"
    page_num_patterns = [
        re.compile(r'^\d+$'),
        re.compile(r'^(page|pg\.?)\s*\|\s*\d+$', re.IGNORECASE),
        re.compile(r'^(page|pg\.?)\s*\d+$', re.IGNORECASE),
        re.compile(r'^\d+\s*[\/\-]\s*\d+$'),
        re.compile(r'^[\-\•]\s*\d+\s*[\-\•]$'), # - 1 -
        re.compile(r'^\d+\s*of\s*\d+$', re.IGNORECASE)
    ]
    
    # 2. Extract Rich Content and Merge
    current_text_item = None
    list_marker_pattern = re.compile(r'^(\d+[\.\)\-]|[a-zA-Z][\.\)\-]|[\•\-\*])\s+')
    
    # [NEW] Pass 1.5: Count Image Frequencies
    # Backgrounds and templates from PPTX repeat exactly on multiple slides.
    import hashlib
    image_hash_counts = {}
    if extract_images:
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 1:
                    img_hash = hashlib.md5(block["image"]).hexdigest()
                    image_hash_counts[img_hash] = image_hash_counts.get(img_hash, 0) + 1
    
    for page_num, page in enumerate(doc):
        page_images = []
        
        # 0. Extract Tables (New Feature)
        # Find tables first so we can process them and avoid extracting their text again as floating blocks
        tables = page.find_tables()
        extracted_tables = []
        table_bboxes = []
        if tables:
            for table in tables:
                # Heuristic: If table covers >80% of the page area, it's likely a false positive (page border)
                table_area = (table.bbox[2] - table.bbox[0]) * (table.bbox[3] - table.bbox[1])
                page_area = page.rect.width * page.rect.height
                if page_area > 0 and (table_area / page_area) > 0.8:
                    continue
                    
                table_bboxes.append(table.bbox)
                
                # Extract table as structured data for proper rendering
                table_rows = []
                for row in table.extract():
                    cleaned_row = [str(cell).replace("\n", " ").strip() if cell else "" for cell in row]
                    table_rows.append(cleaned_row)
                
                if table_rows:
                    img_data = None
                    try:
                        # Capture table image perfectly preserving original formatting
                        table_rect = fitz.Rect(table.bbox)
                        pix = page.get_pixmap(clip=table_rect, dpi=200) # High quality for tables
                        img_data = pix.tobytes("jpeg")
                    except Exception as e:
                        logger.warning(f"Failed to capture table image: {e}")

                    extracted_tables.append({
                        'type': 'table',
                        'rows': table_rows,  # 2D array of strings
                        'bbox': table.bbox,
                        'image_data': img_data
                    })
        
        # Sort tables by vertical position so we can interleave them
        extracted_tables.sort(key=lambda t: t['bbox'][1])
                    
        # 1. Extract Annotations (Highlights)
        # Type 8 is Highlight. We store (rect, color_hex)
        highlights = []
        for annot in page.annots():
             if annot.type[0] == 8: # Highlight
                 # Color is typically RGB float 0-1
                 if annot.colors:
                     stroke = annot.colors.get("stroke") # Some PDF versions differ
                     if not stroke: stroke = annot.colors.get("fill") # fallback
                     
                     if stroke:
                         # Convert to hex
                         r, g, b = [int(str_c * 255) for str_c in stroke] if len(stroke) >= 3 else (255, 255, 0)
                         hex_c = f"#{r:02x}{g:02x}{b:02x}"
                         highlights.append((annot.rect, hex_c))
        
        # Extract dict to get spans and images
        # sort=True ensures blocks are ordered by vertical position, then horizontal
        page_dict = page.get_text("dict", sort=True)
        blocks = page_dict["blocks"]
        
        for block in blocks:
            # We interleave tables by checking vertical coordinates
            block_y0 = block["bbox"][1]
            while extracted_tables and extracted_tables[0]['bbox'][1] <= block_y0 + 5:
                if current_text_item:
                    all_content.append(current_text_item)
                    current_text_item = None
                table_item = extracted_tables.pop(0)
                all_content.append({'type': 'table', 'rows': table_item['rows'], 'page_num': page_num})
                
            b_type = block["type"]
            
            if b_type == 0: # Text
                # Skip text blocks that fall inside any extracted table to avoid duplication
                block_rect = fitz.Rect(block["bbox"])
                in_table = False
                for t_bbox in table_bboxes:
                    # If the center of the block is inside the table bbox, skip it
                    center = block_rect.tl + (block_rect.br - block_rect.tl) * 0.5
                    if fitz.Rect(t_bbox).contains(center):
                        in_table = True
                        break
                
                if in_table:
                    continue

                # Sort lines vertically then horizontally to fix reading order
                # Use int(y/5) to bucket lines that have slightly different y0 but are effectively the same line
                block["lines"].sort(key=lambda l: (int(l["bbox"][1] / 5), l["bbox"][0]))
                
                # Check if this block is excluded
                block_plain_text = ""
                for line in block["lines"]:
                    line["spans"].sort(key=lambda s: s["bbox"][0])
                    for span in line["spans"]:
                        block_plain_text += span["text"] + " "
                
                clean_block_text = " ".join(block_plain_text.splitlines()).strip()
                clean_block_text = re.sub(r'\s+', ' ', clean_block_text).strip()
                
                if not clean_block_text:
                    continue
                
                # 1. Page Number Filter
                is_page_num = False
                if len(clean_block_text) < 20: # Optimization: Page nums are short
                    for pat in page_num_patterns:
                        if pat.match(clean_block_text):
                            is_page_num = True
                            break
                if is_page_num:
                    continue

                # 2. Repetitive Content Filter
                # If text scans as repetitive:
                if clean_block_text in always_remove_texts:
                     # Aggressive removal for very frequent items (likely footers/page nums)
                     continue
                
                if clean_block_text in repetitive_candidate_texts:
                    # If we have seen it before -> Skip (It's a repeated header)
                    if clean_block_text in seen_repetitive_texts:
                        continue
                    else:
                        # First time seeing it -> Keep it (It might be the Title on Page 1)
                        seen_repetitive_texts.add(clean_block_text)
                    
                # Process lines inside this block into semantic items (separating headings and numbered points)
                block_items = []
                current_sub_item = None
                
                for line in block["lines"]:
                    line_html = ""
                    plain_spans = []
                    is_line_bold = False
                    max_span_size = 0.0
                    
                    for span in line["spans"]:
                        text = span["text"]
                        if not text.strip():
                            continue
                        plain_spans.append(text)
                        
                        flags = span["flags"]
                        color = span["color"]
                        bbox = fitz.Rect(span["bbox"])
                        span_size = span.get("size", 10.0)
                        if span_size > max_span_size:
                            max_span_size = span_size
                            
                        span_bold = (flags & 16) > 0 or "bold" in span["font"].lower()
                        if span_bold:
                            is_line_bold = True
                        span_italic = (flags & 2) > 0 or "italic" in span["font"].lower()
                        
                        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        span_html = escaped
                        if span_bold:
                            span_html = f"<b>{span_html}</b>"
                        if span_italic:
                            span_html = f"<i>{span_html}</i>"
                        if color != 0:
                            span_html = f"<font color='#{color:06x}'>{span_html}</font>"
                        for h_rect, h_color in highlights:
                            if bbox.intersects(h_rect):
                                span_html = f"<span backColor='{h_color}'>{span_html}</span>"
                                break
                        line_html += span_html + " "
                    
                    plain_line = " ".join(" ".join(plain_spans).splitlines()).strip()
                    plain_line = re.sub(r'\s+', ' ', plain_line).strip()
                    if not plain_line:
                        continue
                    
                    starts_with_list = bool(list_marker_pattern.match(plain_line))
                    
                    # Detect if line is a Heading
                    is_heading = False
                    if not starts_with_list and len(plain_line) < 130:
                        if not plain_line.endswith('.') or plain_line.endswith(':'):
                            if is_line_bold or max_span_size >= 12.0:
                                is_heading = True
                            elif plain_line.endswith(':') and len(plain_line) < 60:
                                is_heading = True
                                
                    # Check for embedded heading followed by a numbered point:
                    # e.g., "Types of identification 1. Complete/ Total identification: ..."
                    m_embedded = re.search(r'^([A-Za-z\s]{3,50}?:?)\s+(\(?\d+[\.\)\-]\s+[A-Za-z].*)$', plain_line)
                    if m_embedded and not starts_with_list and not plain_line.startswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
                        heading_part = m_embedded.group(1).strip()
                        point_part = m_embedded.group(2).strip()
                        
                        if current_sub_item:
                            block_items.append(current_sub_item)
                            current_sub_item = None
                            
                        block_items.append({
                            'type': 'text',
                            'content': f"<b>{heading_part}</b>",
                            'plain': heading_part,
                            'is_heading': True,
                            'page_num': page_num,
                            'y1': line["bbox"][3]
                        })
                        
                        current_sub_item = {
                            'type': 'text',
                            'content': point_part,
                            'plain': point_part,
                            'is_heading': False,
                            'page_num': page_num,
                            'y1': line["bbox"][3]
                        }
                        continue

                    # If line is a heading
                    if is_heading:
                        if current_sub_item:
                            block_items.append(current_sub_item)
                            current_sub_item = None
                        block_items.append({
                            'type': 'text',
                            'content': f"<b>{plain_line}</b>",
                            'plain': plain_line,
                            'is_heading': True,
                            'page_num': page_num,
                            'y1': line["bbox"][3]
                        })
                        continue

                    # If line starts a numbered/bullet item
                    if starts_with_list:
                        if current_sub_item:
                            block_items.append(current_sub_item)
                            current_sub_item = None
                        current_sub_item = {
                            'type': 'text',
                            'content': line_html.strip(),
                            'plain': plain_line,
                            'is_heading': False,
                            'page_num': page_num,
                            'y1': line["bbox"][3]
                        }
                        continue

                    # Normal continuation line
                    if current_sub_item is None:
                        current_sub_item = {
                            'type': 'text',
                            'content': line_html.strip(),
                            'plain': plain_line,
                            'is_heading': False,
                            'page_num': page_num,
                            'y1': line["bbox"][3]
                        }
                    else:
                        current_sub_item['content'] += " " + line_html.strip()
                        current_sub_item['plain'] += " " + plain_line
                        current_sub_item['y1'] = line["bbox"][3]
                
                if current_sub_item:
                    block_items.append(current_sub_item)
                    
                # Merge or append block_items into all_content
                for sub_item in block_items:
                    # Also check if sub_item contains multiple concatenated numbered points:
                    # e.g., "... 1. Point A. 2. Point B."
                    split_points = re.split(r'(?<=[.!?])\s+(?=\d+[\.\)\-]\s+[A-Za-z])', sub_item['plain'])
                    if len(split_points) > 1:
                        if current_text_item:
                            all_content.append(current_text_item)
                            current_text_item = None
                        for sp in split_points:
                            all_content.append({
                                'type': 'text',
                                'content': sp.strip(),
                                'plain': sp.strip(),
                                'is_heading': False,
                                'page_num': page_num,
                                'y1': sub_item['y1']
                            })
                        continue

                    if current_text_item:
                        prev_plain = current_text_item['plain'].strip()
                        curr_plain = sub_item['plain'].strip()
                        
                        # NEVER merge headings!
                        if current_text_item.get('is_heading') or sub_item.get('is_heading'):
                            all_content.append(current_text_item)
                            current_text_item = sub_item
                            continue
                            
                        # NEVER merge list items into previous items!
                        if list_marker_pattern.match(curr_plain):
                            all_content.append(current_text_item)
                            current_text_item = sub_item
                            continue
                            
                        gap = 9999
                        if current_text_item.get('page_num') == sub_item.get('page_num'):
                            gap = sub_item.get('y1', 0) - current_text_item.get('y1', 0)
                            
                        # Continuation if starts with lowercase and close gap
                        if curr_plain and curr_plain[0].islower() and gap < 40:
                            current_text_item['content'] += " " + sub_item['content']
                            current_text_item['plain'] += " " + sub_item['plain']
                            current_text_item['y1'] = sub_item['y1']
                        else:
                            all_content.append(current_text_item)
                            current_text_item = sub_item
                    else:
                        current_text_item = sub_item
                    
            elif b_type == 1: # Image
                if not extract_images:
                    continue
                try:
                    image_bytes = block["image"]
                    
                    # 1. Check for repeating template graphics
                    # If this image appears on multiple pages, it's almost certainly a background
                    # or Master Slide decoration. We remove it entirely.
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    if image_hash_counts.get(img_hash, 0) > 1:
                        continue
                        
                    # Convert to PIL to get dimensions/verify
                    img = PILImage.open(io.BytesIO(image_bytes))
                    
                    if img.width < 50 or img.height < 50: # Skip very small images/icons
                        continue
                        
                    from PIL import ImageStat
                    # Check for solid colors or extremely smooth gradients
                    # If variance is very low, it's not a meaningful diagram.
                    try:
                        stat = ImageStat.Stat(img)
                        if sum(stat.var) < 150:
                            continue
                    except Exception:
                        pass
                        
                    image_size_kb = len(image_bytes) / 1024.0
                    ratio = img.width / float(img.height) if img.height > 0 else 1.0
                    
                    bbox = block["bbox"]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    page_w = page.rect.width
                    page_h = page.rect.height
                    
                    width_ratio = width / page_w if page_w > 0 else 0
                    height_ratio = height / page_h if page_h > 0 else 0
                    
                    # 1. Very lightweight images (e.g., solid circles, tiny gradients less than 2KB)
                    if image_size_kb < 2.0:
                        continue
                    
                    # (Removed extreme aspect ratio filter as it drops valid wide collages like DNA strips or sample comparisons)
                        
                    # 3. Skip full-bleed backgrounds (images that cover the entire slide)
                    is_full_bleed = width_ratio > 0.95 and height_ratio > 0.95
                    touches_edges = bbox[0] < 10 and bbox[1] < 10
                    if is_full_bleed or (width_ratio > 0.85 and height_ratio > 0.85 and touches_edges):
                        continue
                        
                    # Legitimate image found. We DO NOT flush text here!
                    # We want all text on the slide to stay together so lists aren't broken.

                    # Store as bytes for ReportLab
                    img_byte_arr = io.BytesIO()
                    
                    # Fix PNG transparency rendering as black
                    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                        alpha = img.convert('RGBA').split()[-1]
                        bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
                        bg.paste(img, mask=alpha)
                        bg.convert('RGB').save(img_byte_arr, format='JPEG')
                    else:
                        img.convert('RGB').save(img_byte_arr, format='JPEG')
                        
                    img_data = img_byte_arr.getvalue()
                    
                    page_images.append({
                        'type': 'image',
                        'data': img_data,
                        'width': width,
                        'height': height,
                        'page_num': page_num
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract image on page {page_num}: {e}")
                    
        # Flush remaining tables for this page
        while extracted_tables:
            if current_text_item:
                all_content.append(current_text_item)
                current_text_item = None
            table_item = extracted_tables.pop(0)
            all_content.append({'type': 'table', 'rows': table_item['rows'], 'page_num': page_num})
            
        # Flush text before adding images at the end of the page
        if current_text_item:
            all_content.append(current_text_item)
            current_text_item = None
            
        # Add all images collected for this page at the very end
        all_content.extend(page_images)
    
    # Flush final
    if current_text_item:
        all_content.append(current_text_item)
                    
    return all_content

from pptx.enum.shapes import MSO_SHAPE_TYPE

def extract_text_from_pptx(pptx_file) -> list[dict]:
    """
    Extracts text from a PowerPoint file (.pptx).
    Returns a list of Content Items (currently just Text).
    Handles grouped shapes and tables recursively.
    """
    try:
        if hasattr(pptx_file, 'seek'):
            pptx_file.seek(0)
        
        prs = Presentation(pptx_file)
        text_items = []
        
        def iter_shapes(shapes):
            for shape in shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                    iter_shapes(shape.shapes)
                elif shape.has_text_frame:
                    if shape.text.strip():
                        # Extract each paragraph
                        for paragraph in shape.text_frame.paragraphs:
                            clean_text = paragraph.text.strip()
                            if clean_text:
                                text_items.append({
                                    'type': 'text',
                                    'content': clean_text,
                                    'plain': clean_text
                                })
                elif shape.has_table:
                    # Extract table as structured data
                    table_rows = []
                    for row in shape.table.rows:
                        row_cells = []
                        for cell in row.cells:
                            clean_cell = cell.text_frame.text.replace('\n', ' ').strip() if cell.text_frame else ""
                            row_cells.append(clean_cell)
                        table_rows.append(row_cells)
                    
                    if table_rows:
                        text_items.append({
                            'type': 'table',
                            'rows': table_rows
                        })

        for slide_index, slide in enumerate(prs.slides):
            slide_start = len(text_items)
            iter_shapes(slide.shapes)
            for item in text_items[slide_start:]:
                item['page_num'] = slide_index
                                 
        return text_items
    except Exception as e:
        logger.error(f"Failed to extract text from PPTX: {e}")
        return []

def extract_content(file_obj, file_type, api_key=None, extract_images=True, is_scanned=False, status_text=None, progress_bar=None) -> list[dict]:
    """
    Generic extraction function for PDF and PPTX.
    file_type: 'pdf' or 'pptx'
    """
    if file_type == 'pdf':
        return extract_text_from_pdf(file_obj, api_key=api_key, extract_images=extract_images, is_scanned=is_scanned, status_text=status_text, progress_bar=progress_bar)
    elif file_type == 'pptx':
        return extract_text_from_pptx(file_obj)
    else:
        logger.error(f"Unsupported file type: {file_type}")
        return []

def integrate_translations(content_items: list, api_key: str) -> list:
    """
    Translates all text and table items that don't yet have a translation.
    Items already translated (e.g. from the OCR pass) are skipped.
    Modifies content_items in-place and returns it.
    """
    paragraphs = []
    text_map = []   # (item_index, paragraph_index)
    tables_to_translate = [] # list of (item_index, rows)

    for i, item in enumerate(content_items):
        if item['type'] == 'text':
            if item.get('translation') is not None:
                continue
            text_map.append((i, len(paragraphs)))
            paragraphs.append(item['plain'])
        elif item['type'] == 'table':
            if 'translated_rows' not in item:
                tables_to_translate.append((i, item['rows']))

    # 1. Global Context Generation
    global_context_summary = ""
    if paragraphs:
        sample_text = " ".join(paragraphs[:50]) # Use first 50 paragraphs as context
        if config.USE_DOCUMENT_SUMMARY:
            from translator import generate_document_summary
            global_context_summary = generate_document_summary(sample_text, api_key)

    # Translate text blocks concurrently
    if paragraphs:
        results = list(translate_paragraphs_concurrently(paragraphs, api_key, global_context_summary))
        lookup = {orig: trans for _, orig, trans in results}

        list_pfx_regex = re.compile(r'^(\(?\d+[\.\)\-]\s*|\(?[a-zA-Z][\.\)\-]\s*|[\•\-\*■●▪►]\s*)')
        for item_idx, para_idx in text_map:
            orig_text = paragraphs[para_idx].strip()
            trans_text = lookup.get(paragraphs[para_idx], '').strip()
            
            # Safeguard: If original starts with a number/bullet marker and Arabic lost it, restore it
            m_orig = list_pfx_regex.match(orig_text)
            if m_orig:
                pfx = m_orig.group(1).strip()
                if not re.match(r'^(\(?\d+[\.\)\-]|[\•\-\*■●▪►])', trans_text):
                    trans_text = f"{pfx} {trans_text}"
                    
            content_items[item_idx]['translation'] = trans_text

    # Translate tables intelligently as whole units
    if tables_to_translate:
        import concurrent.futures
        
        def _translate_single_table(tup):
            idx, rows = tup
            return idx, translate_table(rows, api_key, global_context_summary)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(config.MAX_WORKERS, len(tables_to_translate))) as executor:
            future_to_table = {executor.submit(_translate_single_table, t): t for t in tables_to_translate}
            for future in concurrent.futures.as_completed(future_to_table):
                try:
                    idx, translated_rows = future.result()
                    content_items[idx]['translated_rows'] = translated_rows
                except GeminiQuotaExceeded:
                    raise
                except Exception as e:
                    logger.error(f"Error translating table: {e}")
                    idx = future_to_table[future][0]
                    content_items[idx]['translated_rows'] = content_items[idx]['rows']

    return content_items


from reportlab.lib.utils import simpleSplit

def normalize_arabic_text(text: str) -> str:
    """
    Normalizes Persian/Urdu characters to standard Arabic to prevent
    rendering issues (square boxes) with Arabic fonts like Arial that
    lack Persian presentation forms. Also strips zero-width characters.
    """
    if not text:
        return text
        
    replacements = {
        'ی': 'ي',  # Farsi Yeh -> Arabic Yeh
        'ک': 'ك',  # Keheh -> Kaf
        'گ': 'ج',  # Gaf
        'چ': 'ج',  # Cheh
        'پ': 'ب',  # Peh
        'ژ': 'ز',  # Jeh
        'ے': 'ي',  # Yeh Barree
        'ہ': 'ه',  # Heh Goal
        '\u200c': '', # Zero-Width Non-Joiner
        '\u200d': '', # Zero-Width Joiner
        '\u200e': '', # LTR Mark
        '\u200f': '', # RTL Mark
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def process_arabic_text(text, width=None, font_name=config.ARABIC_FONT_NAME, font_size=config.ARABIC_FONT_SIZE, base_dir='R'):
    """
    Reshapes Arabic text and handles BiDi.
    Handles HTML <br/> line separators by splitting on them first,
    reshaping each logical line individually, then rejoining with <br/>.
    This preserves list formatting (bullet points, numbered items) that
    comes from the OCR+Translation pass as <br/>-separated lines.
    """
    if not text:
        return ""
    try:
        text = text.strip()
        if not text:
            return ""

        # Normalize to prevent squares (□) from Farsi characters output by AI
        text = normalize_arabic_text(text)
        
        # [NEW] Strip AI-hallucinated segment IDs (like 0a7, 0b7 at start/end of lines)
        text = re.sub(r'^[0-9a-f][0-9a-f][0-9a-f]\s+', '', text) # Start of string
        text = re.sub(r'\s+[0-9a-f][0-9a-f][0-9a-f]$', '', text) # End of string
        text = re.sub(r'(<br\s*/?>)\s*[0-9a-f][0-9a-f][0-9a-f]\s+', r'\1', text) # After breaks

        # 1. Split on <br/> variants so each logical line is reshaped separately
        #    This preserves list structure that the AI encodes as <br/>-separated items.
        segments = re.split(r'<br\s*/?\s*>', text, flags=re.IGNORECASE)

        reshaped_segments = []
        for seg in segments:
            # 2. Strip remaining HTML tags from this segment (bold, italic, font, etc.)
            plain_seg = re.sub(r'<[^>]+>', ' ', seg)
            plain_seg = re.sub(r'\s+', ' ', plain_seg).strip()
            if not plain_seg:
                continue

            # 3. Reshape Arabic ligatures
            reshaped = arabic_reshaper.reshape(plain_seg)

            if width:
                # 4. Wrap long lines to fit the column width
                lines = simpleSplit(reshaped, font_name, font_size, width)
                # 5. Apply BiDi reversal to each wrapped line
                bidi_lines = [get_display(line.strip(), base_dir=base_dir) for line in lines if line.strip()]
                reshaped_segments.extend(bidi_lines)
            else:
                # No width constraint – just reverse the whole segment
                reshaped_segments.append(get_display(reshaped, base_dir=base_dir))

        # 6. Join all segments with <br/> (valid XHTML for ReportLab)
        return "<br/>".join(reshaped_segments)

    except Exception as e:
        logger.error(f"Error preparing Arabic text: {e}")
        # Last-resort: strip all tags and return plain text
        return re.sub(r'<[^>]+>', '', text)


def _contains_arabic(text):
    """Check if text contains Arabic/Persian characters."""
    for char in text:
        if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\uFB50' <= char <= '\uFDFF' or '\uFE70' <= char <= '\uFEFF':
            return True
    return False

def _sanitize_html_for_reportlab(html_text: str) -> str:
    """
    Fixes HTML markup so ReportLab's XML parser doesn't choke.
    Key fixes:
      - <br>, <br >, <br anything> → <br/>  (must be self-closing in XHTML)
      - </br> → <br/>
      - <b/>, <i/>, <u/> etc. → </b>, </i>, </u>
        (AI models sometimes output self-closing inline tags when they mean closing tags)
    """
    if not html_text:
        return html_text

    # 1. Fix self-closing INLINE tags that should be proper closing tags
    #    e.g. <b/> → </b>, <i/> → </i>
    #    These are NOT valid self-closing XML (only void elements like <br/> are).
    for tag in ('b', 'i', 'u', 'em', 'strong', 'sup', 'sub', 'strike'):
        html_text = re.sub(rf'<{tag}\s*/>', f'</{tag}>', html_text, flags=re.IGNORECASE)

    # 2. Normalise all <br ...> / </br> variants → <br/>
    html_text = re.sub(r'<br\s*/?>', '<br/>', html_text, flags=re.IGNORECASE)
    html_text = re.sub(r'</br\s*>', '<br/>', html_text, flags=re.IGNORECASE)

    # 3. Remove any remaining unclosed/invalid <br ...attribute...> forms
    html_text = re.sub(r'<br\s+[^>]*>', '<br/>', html_text, flags=re.IGNORECASE)

    return html_text


def _wrap_arabic_font(html_text):
    """
    Wraps Arabic portions of HTML content in <font name='Arial'> tags.
    CRITICAL: If the English/Original text contains Arabic characters, ReportLab 
    will render them left-to-right (scrambled) unless we reshape and apply bidi.
    """
    if not html_text:
        return html_text

    # Sanitize first (fixes <br> → <br/> etc.)
    html_text = _sanitize_html_for_reportlab(html_text)

    # Strip tags to check plain text
    plain = re.sub(r'<[^>]+>', '', html_text)
    if not _contains_arabic(plain):
        return html_text
        
    # If it contains Arabic, we MUST reshape it so it renders correctly.
    # We use base_dir='L' because this function is primarily used for the "English" 
    # original text side, so the overall paragraph direction should remain LTR, 
    # but the Arabic substrings will be correctly flipped.
    reshaped_bidi = process_arabic_text(html_text, width=None, base_dir='L')
    
    return f"<font name='{config.ARABIC_FONT_NAME}'>{reshaped_bidi}</font>"


def _merge_staggered_rows(rows):
    """
    Merges consecutive rows that represent vertically-merged cells in the original PDF.

    When PyMuPDF extracts a table with vertically merged cells, it produces multiple
    rows where content alternates between columns. For example:
        ["",               "Long description text"]
        ["Label Name",     ""]
    Becomes:
        ["Label Name",     "Long description text"]

    Also handles label continuations where a label spans multiple rows:
        ["Enzyme",  ""]
        ["Induction", ""]
    Becomes:
        ["Enzyme Induction", ""]
    """
    if not rows or len(rows) < 2:
        return rows

    num_cols = max(len(r) for r in rows)
    # Normalize row lengths
    normalized = []
    for r in rows:
        nr = [str(c).strip() for c in r]
        while len(nr) < num_cols:
            nr.append("")
        normalized.append(nr)

    def is_complete(row):
        """A row where ALL columns have content (e.g. header row)."""
        return all(c.strip() for c in row)

    result = []
    accumulator = None

    for row in normalized:
        if is_complete(row):
            # Flush accumulator if exists
            if accumulator and any(c.strip() for c in accumulator):
                result.append(accumulator)
            # Complete row stands on its own
            result.append(list(row))
            accumulator = None
        else:
            # Partial row — try to merge into accumulator
            if accumulator is None:
                accumulator = [""] * num_cols

            # Find column conflicts (both accumulator and row have content)
            conflicts = [col for col in range(num_cols)
                         if accumulator[col].strip() and row[col].strip()]

            if not conflicts:
                # No conflict → merge freely
                for col in range(num_cols):
                    if row[col].strip() and not accumulator[col].strip():
                        accumulator[col] = row[col]
            else:
                row_filled = sum(1 for c in row if c.strip())
                # Single short-label continuation (e.g. "Enzyme" then "Induction")
                if (row_filled == 1 and len(conflicts) == 1
                        and len(row[conflicts[0]]) < 40
                        and len(accumulator[conflicts[0]]) < 40):
                    col = conflicts[0]
                    accumulator[col] = accumulator[col] + " " + row[col]
                else:
                    # Real conflict → flush and start new group
                    if any(c.strip() for c in accumulator):
                        result.append(accumulator)
                    accumulator = list(row)

    # Flush final accumulator
    if accumulator and any(c.strip() for c in accumulator):
        result.append(accumulator)

    return result


def optimize_table_columns(rows, translated_rows, total_width):
    if not rows:
         return [], [], []
         
    # 0. Ensure strings and strip
    clean_rows = []
    for r in rows:
        clean_rows.append([str(c).strip() for c in r])
    clean_translated = []
    for r in translated_rows:
        clean_translated.append([str(c).strip() for c in r])

    num_cols_rows = max(len(r) for r in clean_rows) if clean_rows else 0
    num_cols_trans = max(len(r) for r in clean_translated) if clean_translated else 0
    num_cols = max(num_cols_rows, num_cols_trans)
    
    if num_cols == 0:
        return clean_rows, clean_translated, []
        
    for r in clean_rows:
        while len(r) < num_cols: r.append("")
    for r in clean_translated:
        while len(r) < num_cols: r.append("")

    # 1a. Merge vertically-staggered rows (from PDF merged cells)
    clean_rows = _merge_staggered_rows(clean_rows)
    clean_translated = _merge_staggered_rows(clean_translated)

    # Re-normalize after row merging (counts may differ)
    num_cols_rows = max(len(r) for r in clean_rows) if clean_rows else 0
    num_cols_trans = max(len(r) for r in clean_translated) if clean_translated else 0
    num_cols = max(num_cols_rows, num_cols_trans)
    if num_cols == 0:
        return clean_rows, clean_translated, []
    for r in clean_rows:
        while len(r) < num_cols: r.append("")
    for r in clean_translated:
        while len(r) < num_cols: r.append("")

    # 1b. Merge staggered / non-colliding columns (horizontal)
    c = 0
    while c < num_cols - 1:
        can_merge = True
        for r_idx in range(len(clean_rows)):
            if clean_rows[r_idx][c] and clean_rows[r_idx][c+1]:
                can_merge = False
                break
        if can_merge:
            for r_idx in range(len(clean_translated)):
                if clean_translated[r_idx][c] and clean_translated[r_idx][c+1]:
                    can_merge = False
                    break
                    
        if can_merge:
            # Merge c+1 into c
            for r in clean_rows:
                r[c] = r[c] if r[c] else r[c+1]
                del r[c+1]
            for r in clean_translated:
                r[c] = r[c] if r[c] else r[c+1]
                del r[c+1]
            num_cols -= 1
        else:
            c += 1

    if num_cols == 0:
        return clean_rows, clean_translated, []

    # 2. Calculate proportional widths based on content length
    #    Use a generous minimum so label columns don't get squished
    col_chars = [0] * num_cols
    for row in clean_rows:
        for i, cell in enumerate(row):
            col_chars[i] = max(col_chars[i], len(cell))
            
    for row in clean_translated:
        for i, cell in enumerate(row):
            col_chars[i] = max(col_chars[i], len(cell))
            
    total_chars = sum(col_chars)
    min_width = min(60, total_width / num_cols)  # Raised from 40 → 60 for better label widths
    
    if total_chars == 0:
        col_widths = [total_width / num_cols] * num_cols
    else:
        remaining = total_width - (min_width * num_cols)
        if remaining < 0:
            col_widths = [total_width / num_cols] * num_cols
        else:
            col_widths = [min_width + (chars / total_chars) * remaining for chars in col_chars]
            
    return clean_rows, clean_translated, col_widths

def generate_dual_language_pdf(content_items, output_path, stacked=False):
    """
    Generates a PDF with English and Arabic content.
    stacked: If True, Arabic translation appears below English in red (single column).
    """
    if not register_fonts():
        logger.error("Could not register fonts. PDF generation might look wrong.")

    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4, 
                                rightMargin=config.PAGE_MARGIN_INCH*inch, leftMargin=config.PAGE_MARGIN_INCH*inch, 
                                topMargin=config.PAGE_MARGIN_INCH*inch, bottomMargin=config.PAGE_MARGIN_INCH*inch)
        story = []
        
        styles = getSampleStyleSheet()
        
        # English Style
        style_en = ParagraphStyle(
            'EnglishStyle',
            parent=styles['Normal'],
            fontName=config.ENGLISH_FONT_NAME,
            fontSize=config.ENGLISH_FONT_SIZE,
            alignment=TA_LEFT,
            leading=16, # Increased from 14 for better readability
            textColor=colors.HexColor('#1F2937'), # Dark gray, not pure black
            spaceAfter=12,
            allowWidows=1,
            allowOrphans=1
        )
        
        # Arabic Style
        style_ar = ParagraphStyle(
            'ArabicStyle',
            parent=styles['Normal'],
            fontName=config.ARABIC_FONT_NAME,
            fontSize=config.ARABIC_FONT_SIZE,
            alignment=TA_RIGHT,
            leading=20, # Increased from 16 for better readability in Arabic
            textColor=colors.HexColor('#1F2937'),
            wordWrap='RTL',
            spaceAfter=12
        )
        
        # Arabic Style for Stacked Mode (Professional Navy Blue)
        style_ar_stacked = ParagraphStyle(
            'ArabicStyleStacked',
            parent=styles['Normal'],
            fontName=config.ARABIC_FONT_NAME,
            fontSize=config.ARABIC_FONT_SIZE,
            alignment=TA_RIGHT,
            leading=16,
            wordWrap='RTL',
            textColor=colors.HexColor('#1F497D'), # Professional Navy Blue
            rightIndent=15, # Indent from right to separate from English
            spaceAfter=12
        )
        
        # ====== STACKED LAYOUT ======
        if stacked:
            full_width = 7.0 * inch
            
            for item in content_items:
                if item['type'] == 'text':
                    en_text = item.get('content', '')
                    ar_text = item.get('translation', '')
                    
                    # English paragraph
                    story.append(Paragraph(_wrap_arabic_font(en_text), style_en))
                    story.append(Spacer(1, 4))
                    
                    # Arabic paragraph (red) below it
                    ar_display = process_arabic_text(ar_text, width=full_width)
                    story.append(Paragraph(ar_display, style_ar_stacked))
                    
                elif item['type'] == 'table':
                    rows = item.get('rows', [])
                    translated_rows = item.get('translated_rows', rows)
                    if not rows:
                        continue
                    
                    # [NEW] If table has image_data, show it first to preserve complex formatting perfectly
                    if item.get('image_data'):
                        try:
                            img_data = item['image_data']
                            img_reader = ImageReader(io.BytesIO(img_data))
                            iw, ih = img_reader.getSize()
                            aspect = ih / float(iw)
                            display_width = min(full_width * 0.9, iw)
                            display_height = display_width * aspect
                            
                            story.append(Spacer(1, 6))
                            story.append(RLImage(io.BytesIO(img_data), width=display_width, height=display_height))
                            
                            caption_style = ParagraphStyle('TableCaption', parent=style_en, fontName='Helvetica-Oblique', fontSize=9, alignment=TA_CENTER)
                            story.append(Paragraph("Original Table Layout (for reference)", caption_style))
                            story.append(Spacer(1, 12))
                        except Exception as e:
                            logger.warning(f"Failed to add table image to PDF: {e}")

                    rows, translated_rows, col_widths = optimize_table_columns(rows, translated_rows, full_width)
                    if not rows:
                        continue
                    num_cols = len(col_widths)
                    
                    # === INLINE BILINGUAL TABLE ===
                    # English + Arabic are rendered INSIDE each cell (not as separate rows).
                    # This preserves the original table structure and avoids doubling the row count.
                    combined_data = []
                    table_styles = [
                        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E5E7EB')),
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1F2937')), # Slate header background
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('LEFTPADDING', (0,0), (-1,-1), 6),
                        ('RIGHTPADDING', (0,0), (-1,-1), 6),
                        ('TOPPADDING', (0,0), (-1,-1), 6),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ]
                    
                    while len(translated_rows) < len(rows):
                        translated_rows.append([""] * num_cols)
                        
                    # Style for Arabic text within each cell (indigo, smaller, right-aligned)
                    style_ar_cell = ParagraphStyle(
                        'ArabicCellStyle',
                        parent=style_ar_stacked,
                        fontSize=config.ARABIC_FONT_SIZE - 1,
                        spaceBefore=3,
                        spaceAfter=0,
                    )
                        
                    for r_idx in range(len(rows)):
                        en_row = rows[r_idx]
                        ar_row = translated_rows[r_idx]
                        
                        merged_cells = []
                        for i in range(num_cols):
                            en_cell_text = en_row[i] if i < len(en_row) else ""
                            ar_cell_text = ar_row[i] if i < len(ar_row) else ""
                            
                            cell_width = col_widths[i] - 8 if i < len(col_widths) else 80
                            
                            # Build cell content: English on top, Arabic (red) below
                            cell_parts = []
                            if en_cell_text.strip():
                                cell_parts.append(Paragraph(_wrap_arabic_font(en_cell_text), style_en))
                            
                            if ar_cell_text.strip():
                                ar_display = process_arabic_text(ar_cell_text, width=cell_width)
                                cell_parts.append(Paragraph(ar_display, style_ar_cell))
                            
                            if cell_parts:
                                # Use a borderless inner table to stack English + Arabic in one cell
                                inner = Table([[p] for p in cell_parts], colWidths=[col_widths[i] - 8 if i < len(col_widths) else None])
                                inner.setStyle(TableStyle([
                                    ('LEFTPADDING', (0,0), (-1,-1), 0),
                                    ('RIGHTPADDING', (0,0), (-1,-1), 0),
                                    ('TOPPADDING', (0,0), (-1,-1), 0),
                                    ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                                ]))
                                merged_cells.append(inner)
                            else:
                                merged_cells.append(Paragraph("", style_en))
                        
                        combined_data.append(merged_cells)
                        
                        # Alternating row background for data rows (skip header r_idx=0 since we styled it in base table_styles)
                        if r_idx > 0:
                            bg = colors.white if r_idx % 2 == 1 else colors.HexColor('#F9FAFB') # Soft gray
                            table_styles.append(('BACKGROUND', (0, r_idx), (-1, r_idx), bg))
                        
                    combined_tbl = Table(combined_data, colWidths=col_widths)
                    combined_tbl.setStyle(TableStyle(table_styles))
                    story.append(combined_tbl)
                    story.append(Spacer(1, 12))
                    
                elif item['type'] == 'image':
                    try:
                        img_data = item['data']
                        available_width = full_width
                        max_height = 300
                        
                        img_reader = ImageReader(io.BytesIO(img_data))
                        iw, ih = img_reader.getSize()
                        aspect = ih / float(iw)
                        
                        orig_w = item.get('width')
                        
                        if orig_w:
                            # Use original width if available, but cap it so it doesn't force a page break unnecessarily
                            display_width = min(available_width * 0.7, orig_w)
                        else:
                            display_width = min(available_width * 0.6, iw)
                            
                        display_height = display_width * aspect
                        
                        if display_height > max_height:
                            scale_factor = max_height / display_height
                            display_height = max_height
                            display_width = display_width * scale_factor
                        
                        rl_img = RLImage(io.BytesIO(img_data), width=display_width, height=display_height)
                        story.append(rl_img)
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        logger.error(f"Failed to add image to PDF: {e}")
            
            doc.build(story)
            return True
        
        # ====== SIDE-BY-SIDE LAYOUT (Classic High-Density Two-Column) ======
        page_w, page_h = A4
        margin_pts = config.PAGE_MARGIN_INCH * inch
        printable_w = page_w - (2 * margin_pts)
        col_width = printable_w / 2.0
        padding_h = 4
        padding_v = 3
        text_width = col_width - (padding_h * 2)
        
        # Table Data
        data = []
        
        # Table Style Base - Clean, compact grid with exact top baseline alignment
        table_style = [
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LINEBELOW', (0,0), (-1,-1), 0.3, colors.HexColor('#E5E7EB')), # Subtle horizontal row separator
            ('LINEBEFORE', (1,0), (1,-1), 0.5, colors.HexColor('#CCCCCC')), # Subtle vertical separator between English & Arabic
            ('LEFTPADDING', (0,0), (-1,-1), padding_h),
            ('RIGHTPADDING', (0,0), (-1,-1), padding_h),
            ('TOPPADDING', (0,0), (-1,-1), padding_v),
            ('BOTTOMPADDING', (0,0), (-1,-1), padding_v),
        ]
        
        style_en_col = ParagraphStyle(
            'EnglishCol', 
            parent=styles['Normal'],
            fontName=config.ENGLISH_FONT_NAME,
            fontSize=config.ENGLISH_FONT_SIZE,
            leading=config.ENGLISH_FONT_SIZE * 1.35,
            alignment=TA_LEFT,
            textColor=colors.black,
            spaceBefore=0,
            spaceAfter=0
        )
        
        style_ar_col = ParagraphStyle(
            'ArabicCol', 
            parent=styles['Normal'],
            fontName=config.ARABIC_FONT_NAME,
            fontSize=config.ARABIC_FONT_SIZE,
            leading=config.ARABIC_FONT_SIZE * 1.55,
            alignment=TA_RIGHT,
            textColor=colors.black,
            wordWrap='RTL',
            spaceBefore=0,
            spaceAfter=0
        )
        
        style_en_heading = ParagraphStyle(
            'EnglishColHeading', 
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=config.ENGLISH_FONT_SIZE + 0.5,
            leading=(config.ENGLISH_FONT_SIZE + 0.5) * 1.35,
            alignment=TA_LEFT,
            textColor=colors.black,
            spaceBefore=1,
            spaceAfter=1
        )
        
        style_ar_heading = ParagraphStyle(
            'ArabicColHeading', 
            parent=styles['Normal'],
            fontName=config.ARABIC_FONT_NAME,
            fontSize=config.ARABIC_FONT_SIZE + 0.5,
            leading=(config.ARABIC_FONT_SIZE + 0.5) * 1.5,
            alignment=TA_RIGHT,
            textColor=colors.black,
            wordWrap='RTL',
            spaceBefore=1,
            spaceAfter=1
        )
        
        row_idx = 0
        
        for item in content_items:
            if item['type'] == 'text':
                en_text = item.get('content', '')
                ar_text = item.get('translation', '')
                
                if not en_text.strip() and not ar_text.strip():
                    continue
                
                if item.get('is_heading'):
                    p_en = Paragraph(_wrap_arabic_font(f"<b>{en_text}</b>"), style_en_heading)
                    ar_display = process_arabic_text(ar_text, width=text_width)
                    p_ar = Paragraph(f"<b>{ar_display}</b>", style_ar_heading)
                    data.append([p_en, p_ar])
                    table_style.append(('BACKGROUND', (0, row_idx), (1, row_idx), colors.HexColor('#F8FAFC')))
                    row_idx += 1
                else:
                    p_en = Paragraph(_wrap_arabic_font(en_text), style_en_col)
                    ar_display = process_arabic_text(ar_text, width=text_width)
                    p_ar = Paragraph(ar_display, style_ar_col)
                    data.append([p_en, p_ar])
                    row_idx += 1
            
            elif item['type'] == 'table':
                rows = item.get('rows', [])
                translated_rows = item.get('translated_rows', rows)
                if not rows:
                    continue
                
                rows, translated_rows, col_widths = optimize_table_columns(rows, translated_rows, col_width - (padding_h * 2))
                if not rows:
                    continue
                
                # Build English sub-table
                en_sub_data = [[Paragraph(_wrap_arabic_font(cell), style_en_col) for cell in row] for row in rows]
                en_sub = Table(en_sub_data, colWidths=col_widths)
                en_sub.setStyle(TableStyle([
                    ('LINEBELOW', (0,0), (-1,-1), 0.3, colors.HexColor('#CCCCCC')),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 2),
                    ('RIGHTPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ]))
                
                # Build Arabic sub-table
                ar_sub_data = []
                for row in translated_rows:
                    ar_row = [Paragraph(process_arabic_text(cell, width=col_widths[i] - 4), style_ar_col) for i, cell in enumerate(row)]
                    ar_sub_data.append(ar_row)
                ar_sub = Table(ar_sub_data, colWidths=col_widths)
                ar_sub.setStyle(TableStyle([
                    ('LINEBELOW', (0,0), (-1,-1), 0.3, colors.HexColor('#CCCCCC')),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 2),
                    ('RIGHTPADDING', (0,0), (-1,-1), 2),
                    ('TOPPADDING', (0,0), (-1,-1), 2),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ]))
                
                data.append([en_sub, ar_sub])
                row_idx += 1
                
            elif item['type'] == 'image':
                try:
                    img_data = item['data']
                    available_width = printable_w
                    max_height = 250
                    
                    img_reader = ImageReader(io.BytesIO(img_data))
                    iw, ih = img_reader.getSize()
                    aspect = ih / float(iw)
                    
                    orig_w = item.get('width')
                    if orig_w:
                        display_width = min(available_width * 0.8, orig_w)
                    else:
                        display_width = min(available_width * 0.7, iw)
                        
                    display_height = display_width * aspect
                    if display_height > max_height:
                        scale_factor = max_height / display_height
                        display_height = max_height
                        display_width = display_width * scale_factor
                    
                    rl_img = RLImage(io.BytesIO(img_data), width=display_width, height=display_height)
                    
                    data.append([rl_img, ''])
                    table_style.append(('SPAN', (0, row_idx), (1, row_idx)))
                    table_style.append(('ALIGN', (0, row_idx), (1, row_idx), 'CENTER'))
                    row_idx += 1
                except Exception as e:
                    logger.error(f"Failed to add image to PDF: {e}")
        
        if not data:
            data.append([Paragraph("No content extracted", style_en_col), Paragraph("", style_ar_col)])

        t = Table(data, colWidths=[col_width, col_width], repeatRows=0)
        t.setStyle(TableStyle(table_style))
        
        story.append(t)
        doc.build(story)
        return True
    
    except Exception as e:
        logger.error(f"Error building PDF: {e}")
        return False

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_DIRECTION


def remove_control_characters(s):
    if not isinstance(s, str):
        return str(s)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)

def add_rich_text_to_paragraph(paragraph, html_text):
    """
    Parses simple HTML (<b>, <i>, & entities) and adds runs to a docx paragraph.
    """
    if not html_text:
        return

    # Regex to capture tags and entities
    # Matches: <b>, </b>, <i>, </i>, <font ...>, </font>, &...;
    tokens = re.split(r'(<[^>]+>|&[a-z]+;)', html_text)
    
    is_bold = False
    is_italic = False
    
    for token in tokens:
        if not token:
            continue
            
        if token == '<b>':
            is_bold = True
        elif token == '</b>':
            is_bold = False
        elif token == '<i>':
            is_italic = True
        elif token == '</i>':
            is_italic = False
        elif token.startswith('<font') or token == '</font>' or token.startswith('<span') or token == '</span>':
            # Skip font/span tags for now (color support can be added later)
            continue
        elif token == '&lt;':
            run = paragraph.add_run('<')
            run.bold = is_bold
            run.italic = is_italic
        elif token == '&gt;':
            run = paragraph.add_run('>')
            run.bold = is_bold
            run.italic = is_italic
        elif token == '&amp;':
            run = paragraph.add_run('&')
            run.bold = is_bold
            run.italic = is_italic
        elif token.startswith('<'):
            # Other tags ignored
            continue
        else:
            # Text content
            run = paragraph.add_run(remove_control_characters(token))
            run.bold = is_bold
            run.italic = is_italic
            run.font.name = "Arial" if not is_italic else None # Keep base font or let style decide

def generate_dual_language_docx(content_items, output_path, stacked=False):
    """
    Generates a Word Document (.docx) with English and Arabic content.
    stacked: If True, Arabic translation appears below English in red (no table).
    """
    try:
        from docx.shared import RGBColor
        doc = Document()
        
        for section in doc.sections:
            section.top_margin = Inches(0.4)
            section.bottom_margin = Inches(0.4)
            section.left_margin = Inches(0.4)
            section.right_margin = Inches(0.4)

        # ====== STACKED LAYOUT ======
        if stacked:
            for item in content_items:
                if item['type'] == 'text':
                    # English paragraph
                    p_en = doc.add_paragraph()
                    add_rich_text_to_paragraph(p_en, item.get('content', ''))
                    
                    # Arabic paragraph (red)
                    ar_text = item.get('translation', '')
                    p_ar = doc.add_paragraph()
                    p_ar.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
                    run_ar = p_ar.add_run(remove_control_characters(ar_text))
                    run_ar.font.name = "Arial"
                    run_ar.font.size = Pt(12)
                    run_ar.font.rtl = True
                    run_ar.font.color.rgb = RGBColor(0xFF, 0x00, 0x00)  # Red
                    
                elif item['type'] == 'table':
                    rows = item.get('rows', [])
                    translated_rows = item.get('translated_rows', rows)
                    if not rows:
                        continue
                    
                    if item.get('image_data'):
                        try:
                            p_tbl_img = doc.add_paragraph()
                            p_tbl_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                            img_stream = io.BytesIO(item['image_data'])
                            with PILImage.open(io.BytesIO(item['image_data'])) as img:
                                w_in = min(6.0, img.width / 96.0)
                            p_tbl_img.add_run().add_picture(img_stream, width=Inches(w_in))
                            
                            p_caption = doc.add_paragraph()
                            p_caption.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                            run_caption = p_caption.add_run("Original Table Layout (for reference)")
                            run_caption.font.italic = True
                            run_caption.font.size = Pt(9)
                        except Exception as e:
                            pass

                    rows, translated_rows, _ = optimize_table_columns(rows, translated_rows, 100)
                    if not rows:
                        continue
                    num_cols = len(rows[0]) if rows else 0
                    
                    # === SINGLE UNIFIED TABLE (English + Arabic in each cell) ===
                    doc.add_paragraph('', style='Normal')  # Small spacer
                    unified_tbl = doc.add_table(rows=len(rows), cols=num_cols)
                    unified_tbl.style = 'Table Grid'
                    
                    while len(translated_rows) < len(rows):
                        translated_rows.append([""] * num_cols)
                    
                    for r_idx, row in enumerate(rows):
                        for c_idx, cell in enumerate(row):
                            if c_idx < len(unified_tbl.rows[r_idx].cells):
                                cell_obj = unified_tbl.rows[r_idx].cells[c_idx]
                                
                                # English text
                                p_en = cell_obj.paragraphs[0]
                                add_rich_text_to_paragraph(p_en, cell)
                                
                                # Arabic translation (red, below English)
                                ar_cell_text = translated_rows[r_idx][c_idx] if c_idx < len(translated_rows[r_idx]) else ""
                                if ar_cell_text.strip():
                                    p_ar = cell_obj.add_paragraph()
                                    p_ar.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
                                    clean_text = remove_control_characters(re.sub(r'<[^>]+>', '', ar_cell_text))
                                    run = p_ar.add_run(clean_text)
                                    run.font.name = 'Arial'
                                    run.font.size = Pt(10)
                                    run.font.rtl = True
                                    run.font.color.rgb = RGBColor(0xFF, 0x00, 0x00)
                    
                elif item['type'] == 'image':
                    try:
                        img_data = item['data']
                        p_img = doc.add_paragraph()
                        p_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                        image_stream = io.BytesIO(img_data)
                        orig_w = item.get('width')
                        with PILImage.open(io.BytesIO(img_data)) as img:
                            if orig_w:
                                w_in = min(6.0, orig_w / 72.0)
                            else:
                                w_in = min(5.0, img.width / 96.0) # slightly reduced default
                        p_img.add_run().add_picture(image_stream, width=Inches(w_in))
                    except Exception as e:
                        logger.warning(f"Failed to add image to DOCX: {e}")
            
            doc.save(output_path)
            return True
        
        # ====== SIDE-BY-SIDE LAYOUT (Classic High-Density Two-Column) ======
        table = doc.add_table(rows=0, cols=2)
        table.style = 'Table Grid'
        
        for item in content_items:
            if item['type'] == 'text':
                en_text = item.get('content', '')
                ar_text = item.get('translation', '')
                if not en_text.strip() and not ar_text.strip():
                    continue
                
                row_cells = table.add_row().cells
                
                # English Cell (Rich Text)
                is_h = item.get('is_heading', False)
                p_en = row_cells[0].paragraphs[0]
                p_en.paragraph_format.space_before = Pt(3 if is_h else 1)
                p_en.paragraph_format.space_after = Pt(2 if is_h else 1)
                add_rich_text_to_paragraph(p_en, f"<b>{en_text}</b>" if is_h and not en_text.startswith("<b>") else en_text)
                for r in p_en.runs:
                    r.font.size = Pt(10.5 if is_h else 10)
                    if is_h:
                        r.font.bold = True
                
                # Arabic Cell
                p_ar = row_cells[1].paragraphs[0]
                p_ar.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
                p_ar.paragraph_format.space_before = Pt(3 if is_h else 1)
                p_ar.paragraph_format.space_after = Pt(2 if is_h else 1)
                run_ar = p_ar.add_run(remove_control_characters(ar_text))
                run_ar.font.name = "Arial"
                run_ar.font.size = Pt(10.5 if is_h else 10)
                run_ar.font.rtl = True
                if is_h:
                    run_ar.font.bold = True
                
            elif item['type'] == 'table':
                rows = item.get('rows', [])
                translated_rows = item.get('translated_rows', rows)
                if not rows:
                    continue
                
                if item.get('image_data'):
                    try:
                        row = table.add_row()
                        cell = row.cells[0]
                        cell.merge(row.cells[1])
                        p_img = cell.paragraphs[0]
                        p_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                        img_stream = io.BytesIO(item['image_data'])
                        with PILImage.open(io.BytesIO(item['image_data'])) as img:
                            w_in = min(6.0, img.width / 96.0)
                        p_img.add_run().add_picture(img_stream, width=Inches(w_in))
                        p_caption = cell.add_paragraph()
                        p_caption.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                        run_caption = p_caption.add_run("Original Table Layout")
                        run_caption.font.italic = True
                    except Exception as e:
                        pass

                rows, translated_rows, _ = optimize_table_columns(rows, translated_rows, 100)
                if not rows:
                    continue
                num_cols = len(rows[0]) if rows else 0
                row_cells = table.add_row().cells
                
                # English sub-table in left cell
                row_cells[0].text = "" # Clear default text
                en_tbl_inner = row_cells[0].add_table(rows=len(rows), cols=num_cols)
                en_tbl_inner.style = 'Table Grid'
                for r_idx, row_data in enumerate(rows):
                    for c_idx, cell_data in enumerate(row_data):
                        if c_idx < len(en_tbl_inner.rows[r_idx].cells):
                            p = en_tbl_inner.rows[r_idx].cells[c_idx].paragraphs[0]
                            add_rich_text_to_paragraph(p, cell_data)
                
                # Arabic sub-table in right cell
                row_cells[1].text = ""
                ar_tbl_inner = row_cells[1].add_table(rows=len(translated_rows), cols=num_cols)
                ar_tbl_inner.style = 'Table Grid'
                for r_idx, row_data in enumerate(translated_rows):
                    for c_idx, cell_data in enumerate(row_data):
                        if c_idx < len(ar_tbl_inner.rows[r_idx].cells):
                            p = ar_tbl_inner.rows[r_idx].cells[c_idx].paragraphs[0]
                            p.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
                            clean_text = remove_control_characters(re.sub(r'<[^>]+>', '', cell_data))
                            run = p.add_run(clean_text)
                            run.font.name = 'Arial'
                            run.font.size = Pt(11)
                            run.font.rtl = True
                
            elif item['type'] == 'image':
                try:
                    img_data = item['data']
                    row = table.add_row()
                    cell = row.cells[0]
                    cell.merge(row.cells[1])
                    
                    p_img = cell.paragraphs[0]
                    p_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
                    image_stream = io.BytesIO(img_data)
                    orig_w = item.get('width')
                    with PILImage.open(io.BytesIO(img_data)) as img:
                        if orig_w:
                            w_in = min(6.0, orig_w / 72.0)
                        else:
                            w_in = min(5.0, img.width / 96.0)
                    p_img.add_run().add_picture(image_stream, width=Inches(w_in))
                    
                except Exception as e:
                    logger.warning(f"Failed to add image to DOCX: {e}")

        doc.save(output_path)
        return True
    except Exception as e:
        logger.error(f"Error building DOCX: {e}")
        return False



def generate_documents(content_items, base_filename="translated_document", stacked=False):
    """
    Generates both PDF and DOCX from content items.
    Returns (pdf_path, docx_path).
    """
    import concurrent.futures
    
    def _gen_pdf():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            if generate_dual_language_pdf(content_items, tmp_pdf.name, stacked=stacked):
                return tmp_pdf.name
        return None
        
    def _gen_docx():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
            if generate_dual_language_docx(content_items, tmp_docx.name, stacked=stacked):
                return tmp_docx.name
        return None
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_pdf = executor.submit(_gen_pdf)
        future_docx = executor.submit(_gen_docx)
        
        pdf_path = future_pdf.result()
        docx_path = future_docx.result()
            
    return pdf_path, docx_path


# ============================================================
#  SLIDE TRANSLATION MODE
# ============================================================

def _parse_hex_color(hex_str: str):
    """Parses a hex color string like '#cc2222' into an (R, G, B) tuple."""
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) == 3:
        hex_str = ''.join(c * 2 for c in hex_str)
    if len(hex_str) != 6:
        return (0, 0, 0)
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def _draw_wrapped_text_pil(draw, text, font, x, y, max_width, fill, line_height, align='left'):
    """
    Draws text line by line with word-wrapping using Pillow ImageDraw.
    Returns the y position after the last drawn line.
    align: 'left' or 'right'
    """
    if not text:
        return y

    lines = []
    # Split on existing newline markers
    for raw_line in text.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            lines.append('')
            continue
        words = raw_line.split(' ')
        current = ''
        for word in words:
            test = (current + ' ' + word).strip()
            try:
                bbox = font.getbbox(test)
                w = bbox[2] - bbox[0]
            except Exception:
                w = len(test) * 7
            if w <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

    for line in lines:
        if not line:
            y += int(line_height * 0.5)
            continue
        try:
            bbox = font.getbbox(line)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(line) * 7

        if align == 'right':
            draw_x = x + max_width - text_w
        else:
            draw_x = x

        draw.text((draw_x, y), line, font=font, fill=fill)
        y += line_height

    return y


def _fit_ar_text_in_box(draw, text_ar, font_path, box_x, box_y, box_w, box_h,
                        fill_color, align, start_size):
    """
    Draws Arabic text inside a pixel bounding box.
    Auto-shrinks font to fit. Returns True if something was drawn.
    Strategy: reshape+bidi is applied PER DISPLAY LINE (after word-wrap) so that
    RTL text renders correctly and doesn't get clipped.
    """
    from PIL import ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    if not text_ar.strip():
        return False

    INNER_PAD = 3
    usable_w  = max(10, box_w - INNER_PAD * 2)
    usable_h  = max(8,  box_h - INNER_PAD)

    def _measure(font, txt):
        try:
            bb = font.getbbox(txt)
            return bb[2] - bb[0]
        except Exception:
            return len(txt) * max(5, font.size // 2 if hasattr(font, 'size') else 6)

    def _wrap_arabic(raw_text, font, max_w):
        """Word-wraps raw Arabic text, then reshapes+bidi each line independently."""
        result_lines = []
        for para in raw_text.split('\n'):
            para = para.strip()
            if not para:
                result_lines.append('')
                continue
            words = para.split(' ')
            cur_words = []
            for word in words:
                test_words = cur_words + [word]
                # Build test string, reshape+bidi to measure display width accurately
                test_raw = ' '.join(test_words)
                try:
                    test_disp = get_display(arabic_reshaper.reshape(test_raw), base_dir='R')
                except Exception:
                    test_disp = test_raw
                if _measure(font, test_disp) <= max_w:
                    cur_words.append(word)
                else:
                    if cur_words:
                        raw_line = ' '.join(cur_words)
                        try:
                            disp_line = get_display(arabic_reshaper.reshape(raw_line), base_dir='R')
                        except Exception:
                            disp_line = raw_line
                        result_lines.append(disp_line)
                    cur_words = [word]
            if cur_words:
                raw_line = ' '.join(cur_words)
                try:
                    disp_line = get_display(arabic_reshaper.reshape(raw_line), base_dir='R')
                except Exception:
                    disp_line = raw_line
                result_lines.append(disp_line)
            result_lines.append('')  # paragraph gap
        while result_lines and not result_lines[-1]:
            result_lines.pop()
        return result_lines

    for size in range(min(start_size, usable_h), 5, -2):
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()

        line_h = int(size * 1.35)
        lines  = _wrap_arabic(text_ar, font, usable_w)
        total_h = sum(line_h if ln else line_h // 2 for ln in lines)

        if total_h <= usable_h or size <= 7:
            y = box_y + INNER_PAD
            for line in lines:
                if not line:
                    y += line_h // 2
                    continue
                lw = _measure(font, line)
                if align == 'center':
                    x = box_x + INNER_PAD + max(0, (usable_w - lw) // 2)
                elif align == 'right':
                    x = box_x + INNER_PAD + max(0, usable_w - lw)
                else:
                    x = box_x + INNER_PAD
                draw.text((x, y), line, font=font, fill=fill_color)
                y += line_h
                if y >= box_y + box_h - 2:
                    break
            return True

    return False


# DPI used when rendering PDF pages for slide translation.
# Higher = sharper text, better AI positioning accuracy.
SLIDE_RENDER_DPI = 200


def _extract_block_fields(block, slide_bg_hex):
    """Unpacks a block (dict or Pydantic object) into a plain tuple."""
    if isinstance(block, dict):
        return (
            block.get('text_ar', ''),
            float(block.get('x_pct', 0)),  float(block.get('y_pct', 0)),
            float(block.get('w_pct', 10)), float(block.get('h_pct', 5)),
            block.get('font_color_hex', '#000000'),
            block.get('bg_color_hex', slide_bg_hex),
            int(block.get('font_size_pt', 12)),
            block.get('text_align', 'right'),
        )
    return (
        getattr(block, 'text_ar', ''),
        float(getattr(block, 'x_pct', 0)),  float(getattr(block, 'y_pct', 0)),
        float(getattr(block, 'w_pct', 10)), float(getattr(block, 'h_pct', 5)),
        getattr(block, 'font_color_hex', '#000000'),
        getattr(block, 'bg_color_hex', slide_bg_hex),
        int(getattr(block, 'font_size_pt', 12)),
        getattr(block, 'text_align', 'right'),
    )


def _apply_arabic_blocks(draw, orig_img, blocks, slide_bg_hex):
    """
    Erases each original text bounding box and draws the Arabic translation in its place.
    Works on a Pillow ImageDraw attached to a copy of the slide image.
    """
    orig_w, orig_h = orig_img.size
    # Scale AI point-size estimates to pixels at SLIDE_RENDER_DPI
    DPI_SCALE = SLIDE_RENDER_DPI / 72.0

    for block in blocks:
        text_ar, x_pct, y_pct, w_pct, h_pct, color_hex, bg_hex, font_size, align = \
            _extract_block_fields(block, slide_bg_hex)

        if not text_ar.strip():
            continue

        x1 = max(0, int(x_pct / 100.0 * orig_w))
        y1 = max(0, int(y_pct / 100.0 * orig_h))
        x2 = min(orig_w, int((x_pct + w_pct) / 100.0 * orig_w))
        y2 = min(orig_h, int((y_pct + h_pct) / 100.0 * orig_h))
        box_w = max(10, x2 - x1)
        box_h = max(8,  y2 - y1)

        # ── Erase original text ──────────────────────────────────
        bg_rgb = _parse_hex_color(bg_hex)
        # Sample the actual background from a wider ring around the box
        # for a more accurate fill color on gradient / textured slides.
        try:
            pad = max(5, min(12, min(box_w, box_h) // 4))
            sample_pts = [
                (max(0, x1 - pad), max(0, y1 - pad)),
                (min(orig_w-1, x2 + pad), max(0, y1 - pad)),
                (max(0, x1 - pad), min(orig_h-1, y2 + pad)),
                (min(orig_w-1, x2 + pad), min(orig_h-1, y2 + pad)),
                # Mid-edge samples for better accuracy on gradients
                ((x1 + x2) // 2, max(0, y1 - pad)),
                ((x1 + x2) // 2, min(orig_h-1, y2 + pad)),
            ]
            sampled = [orig_img.getpixel(pt)[:3] for pt in sample_pts]
            r = sum(c[0] for c in sampled) // len(sampled)
            g = sum(c[1] for c in sampled) // len(sampled)
            b = sum(c[2] for c in sampled) // len(sampled)
            ai_bright  = 0.299*bg_rgb[0] + 0.587*bg_rgb[1] + 0.114*bg_rgb[2]
            smp_bright = 0.299*r + 0.587*g + 0.114*b
            if abs(ai_bright - smp_bright) < 80:
                bg_rgb = (r, g, b)
        except Exception:
            pass

        draw.rectangle([x1, y1, x2, y2], fill=bg_rgb)

        # ── Resolve text color (ensure readable contrast) ────────
        text_rgb = _parse_hex_color(color_hex)
        brightness    = 0.299*text_rgb[0] + 0.587*text_rgb[1] + 0.114*text_rgb[2]
        bg_brightness = 0.299*bg_rgb[0]   + 0.587*bg_rgb[1]   + 0.114*bg_rgb[2]
        if abs(brightness - bg_brightness) < 60:
            text_rgb = (20, 20, 20) if bg_brightness > 127 else (240, 240, 240)

        ar_align = 'right' if align != 'center' else 'center'

        font_size_px = min(
            max(10, int(font_size * DPI_SCALE)),  # AI size scaled to DPI
            max(10, int(box_h * 0.75))            # never taller than 75% of box
        )
        _fit_ar_text_in_box(draw, text_ar, config.ARABIC_FONT_FILE,
                            x1, y1, box_w, box_h, text_rgb, ar_align, font_size_px)


def _draw_slide_header(canvas, left_w, divider_w, right_start, header_h, lbl_font,
                       left_label, right_label):
    """
    Draws a two-tone professional header bar across the full canvas width.
    Left half: dark navy.  Right half: dark teal.
    """
    from PIL import ImageDraw
    import arabic_reshaper
    from bidi.algorithm import get_display

    total_w = canvas.size[0]
    draw = ImageDraw.Draw(canvas)

    LEFT_BG  = (15,  52,  96)   # dark navy  — original side
    RIGHT_BG = (10,  75,  60)   # dark teal  — Arabic side
    DIVIDER  = (80, 100, 120)
    TXT_CLR  = (255, 255, 255)

    draw.rectangle([0,                    0, left_w - 1,    header_h - 1], fill=LEFT_BG)
    draw.rectangle([left_w,               0, left_w + divider_w - 1, header_h - 1], fill=DIVIDER)
    draw.rectangle([right_start,          0, total_w - 1,   header_h - 1], fill=RIGHT_BG)

    def _centered_text(text, region_x, region_w):
        try:
            tw = lbl_font.getbbox(text)[2]
        except Exception:
            tw = max(60, len(text) * 9)
        x = region_x + (region_w - tw) // 2
        y = (header_h - 18) // 2
        draw.text((x, y), text, font=lbl_font, fill=TXT_CLR)

    try:
        l_shaped = get_display(arabic_reshaper.reshape(left_label))
    except Exception:
        l_shaped = left_label
    try:
        r_shaped = get_display(arabic_reshaper.reshape(right_label))
    except Exception:
        r_shaped = right_label

    _centered_text(l_shaped, 0, left_w)
    _centered_text(r_shaped, right_start, left_w)


def render_bilingual_slide(original_image_bytes: bytes, translation_data: dict) -> bytes:
    """
    Produces a side-by-side mirror image with a professional header bar:
      Header — left: "الأصل (English)" | right: "الترجمة العربية"
      Left half  — original slide, pixel-perfect, untouched
      Right half — same slide with text erased and replaced by Arabic translation

    translation_data: dict from translate_slide_positioned()
      {'slide_bg_color': '#hex', 'blocks': [{x_pct, y_pct, w_pct, h_pct, ...}]}
    Returns PNG bytes.
    """
    from PIL import ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    DIVIDER_W  = 6
    HEADER_H   = 42
    LABEL_SIZE = 18

    # ── Open original ──────────────────────────────────────────────
    try:
        orig = PILImage.open(io.BytesIO(original_image_bytes)).convert('RGB')
    except Exception as e:
        logger.error(f"render_bilingual_slide: cannot open image: {e}")
        fallback = PILImage.new('RGB', (1200, 700), (230, 230, 230))
        buf = io.BytesIO(); fallback.save(buf, format='PNG'); return buf.getvalue()

    orig_w, orig_h = orig.size
    canvas_w = orig_w * 2 + DIVIDER_W
    canvas_h = orig_h + HEADER_H

    canvas = PILImage.new('RGB', (canvas_w, canvas_h), (235, 235, 235))

    # ── Build Arabic-translated right slide ────────────────────────
    right = orig.copy()
    draw  = ImageDraw.Draw(right)

    slide_bg_hex = translation_data.get('slide_bg_color', '#ffffff') \
        if isinstance(translation_data, dict) else '#ffffff'
    blocks = translation_data.get('blocks', []) if isinstance(translation_data, dict) else []

    _apply_arabic_blocks(draw, orig, blocks, slide_bg_hex)

    # ── Compose canvas ─────────────────────────────────────────────
    canvas.paste(orig,  (0,                 HEADER_H))
    canvas.paste(right, (orig_w + DIVIDER_W, HEADER_H))

    # Vertical divider strip (below header)
    div_img = PILImage.new('RGB', (DIVIDER_W, orig_h), (100, 120, 140))
    canvas.paste(div_img, (orig_w, HEADER_H))

    # ── Professional header bar ────────────────────────────────────
    try:
        lbl_font = ImageFont.truetype(config.ARABIC_FONT_FILE, LABEL_SIZE)
    except Exception:
        lbl_font = ImageFont.load_default()

    _draw_slide_header(canvas, orig_w, DIVIDER_W, orig_w + DIVIDER_W,
                       HEADER_H, lbl_font,
                       left_label="الأصل (English)",
                       right_label="الترجمة العربية")

    buf = io.BytesIO()
    canvas.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def render_arabic_only_slide(original_image_bytes: bytes, translation_data: dict) -> bytes:
    """
    Produces a FULL-SIZE Arabic translation slide — same dimensions as the original.
    Each text block is erased and replaced with the Arabic translation.
    A professional header strip is added at the top.

    This is much easier to read than the side-by-side layout because the text
    is never squished into half the width.
    Returns PNG bytes.
    """
    from PIL import ImageDraw, ImageFont
    import arabic_reshaper
    from bidi.algorithm import get_display

    HEADER_H   = 42
    HEADER_BG  = (15, 52, 96)   # dark navy
    LABEL_SIZE = 18
    TXT_CLR    = (255, 255, 255)

    # ── Open original ──────────────────────────────────────────────
    try:
        orig = PILImage.open(io.BytesIO(original_image_bytes)).convert('RGB')
    except Exception as e:
        logger.error(f"render_arabic_only_slide: cannot open image: {e}")
        fallback = PILImage.new('RGB', (1200, 700), (230, 230, 230))
        buf = io.BytesIO(); fallback.save(buf, format='PNG'); return buf.getvalue()

    orig_w, orig_h = orig.size
    canvas = PILImage.new('RGB', (orig_w, orig_h + HEADER_H), (255, 255, 255))

    # ── Build Arabic slide ─────────────────────────────────────────
    ar_slide = orig.copy()
    draw = ImageDraw.Draw(ar_slide)

    slide_bg_hex = translation_data.get('slide_bg_color', '#ffffff') \
        if isinstance(translation_data, dict) else '#ffffff'
    blocks = translation_data.get('blocks', []) if isinstance(translation_data, dict) else []

    _apply_arabic_blocks(draw, orig, blocks, slide_bg_hex)
    canvas.paste(ar_slide, (0, HEADER_H))

    # ── Header bar ────────────────────────────────────────────────
    header_draw = ImageDraw.Draw(canvas)
    header_draw.rectangle([0, 0, orig_w - 1, HEADER_H - 1], fill=HEADER_BG)

    try:
        lbl_font = ImageFont.truetype(config.ARABIC_FONT_FILE, LABEL_SIZE)
    except Exception:
        lbl_font = ImageFont.load_default()

    try:
        label = get_display(arabic_reshaper.reshape("الترجمة العربية"))
        tw = lbl_font.getbbox(label)[2]
    except Exception:
        label, tw = "Arabic Translation", 180

    lx = (orig_w - tw) // 2
    ly = (HEADER_H - LABEL_SIZE) // 2
    header_draw.text((lx, ly), label, font=lbl_font, fill=TXT_CLR)

    buf = io.BytesIO()
    canvas.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def images_to_pdf(png_images: list, output_path: str, fit_a4_landscape: bool = False) -> bool:
    """
    Converts a list of PNG byte strings into a single PDF file.
    If fit_a4_landscape is True, each image is scaled to fill an A4 Landscape page
    as much as possible, picking the best orientation (landscape/portrait) based on
    the image aspect ratio to avoid large empty regions.
    """
    try:
        pil_images = []
        for png_bytes in png_images:
            img = PILImage.open(io.BytesIO(png_bytes)).convert('RGB')
            
            if fit_a4_landscape:
                img_w, img_h = img.size

                # Always produce A4 Landscape pages at 150 DPI (297×210 mm)
                A4_LAND_W, A4_LAND_H = 1754, 1240
                canvas_w, canvas_h = A4_LAND_W, A4_LAND_H
                
                canvas = PILImage.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
                
                scale = min(canvas_w / img_w, canvas_h / img_h)
                new_w = max(1, int(img_w * scale))
                new_h = max(1, int(img_h * scale))
                
                # Use Lanczos resampling for best quality
                if hasattr(PILImage, 'Resampling'):
                    resample_filter = PILImage.Resampling.LANCZOS
                else:
                    resample_filter = PILImage.LANCZOS
                    
                img_resized = img.resize((new_w, new_h), resample_filter)
                
                # Paste centered
                x = (canvas_w - new_w) // 2
                y = (canvas_h - new_h) // 2
                canvas.paste(img_resized, (x, y))
                pil_images.append(canvas)
            else:
                pil_images.append(img)

        if not pil_images:
            return False

        first = pil_images[0]
        rest  = pil_images[1:] if len(pil_images) > 1 else []
        first.save(output_path, format='PDF', save_all=True, append_images=rest,
                   resolution=SLIDE_RENDER_DPI)
        return True
    except Exception as e:
        logger.error(f"images_to_pdf failed: {e}")
        return False


def process_as_slides(file_obj, file_type: str, api_key: str,
                      status_text=None, progress_bar=None,
                      layout: str = 'side_by_side') -> str | None:
    """
    Slide Translation Mode.

    layout options:
      'side_by_side'  — original left | Arabic right (comparison view)
      'arabic_only'   — full-size Arabic slide only (best for reading)

    Steps:
    1. Render each PDF page as a JPEG at SLIDE_RENDER_DPI.
    2. Send each image to AI → positioned bilingual text blocks.
    3. Compose output image per slide according to layout.
    4. Bundle all images into a single output PDF.
    Returns the path to the generated PDF, or None on failure.
    """
    import concurrent.futures
    from translator import translate_slide_positioned

    # ── Open document ─────────────────────────────────────────────
    try:
        if isinstance(file_obj, str):
            doc = fitz.open(file_obj)
        else:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            doc = fitz.open(stream=file_obj.read(), filetype='pdf')
    except Exception as e:
        logger.error(f"process_as_slides: cannot open file: {e}")
        return None

    total_pages = len(doc)
    if total_pages == 0:
        return None

    layout_label = "جانبي ثنائي اللغة" if layout == 'side_by_side' else "عربي كامل"
    if status_text:
        status_text.info(f"🎨 وضع السلايدات ({layout_label}): معالجة {total_pages} شريحة...")
    if progress_bar:
        progress_bar.progress(5)

    # ── Step 1: Render all pages to JPEG bytes ────────────────────
    def _render_page(page_num):
        page = doc[page_num]
        pix  = page.get_pixmap(dpi=SLIDE_RENDER_DPI)
        return page_num, pix.tobytes('jpeg')

    rendered_pages = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_render_page, i): i for i in range(total_pages)}
        for fut in concurrent.futures.as_completed(futs):
            pn, img_bytes = fut.result()
            rendered_pages[pn] = img_bytes

    if progress_bar:
        progress_bar.progress(8)

    # ── Step 2a: Extract positioned text blocks from all slides (Gemini) ──
    from translator import extract_slide_blocks, translate_texts_gemini

    if status_text:
        status_text.info(f"🔍 استخراج نصوص الشرائح... (0/{total_pages})")

    extracted_pages = {}
    completed = 0

    def _compress_for_ai(img_bytes: bytes, max_width: int = 1280, quality: int = 72) -> bytes:
        """Resize + recompress a JPEG so Gemini doesn't drop the connection on large payloads."""
        img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = img.size
        if w > max_width:
            img = img.resize((max_width, int(h * max_width / w)),
                             PILImage.Resampling.LANCZOS if hasattr(PILImage, 'Resampling') else PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        return buf.getvalue()

    def _extract_page(args):
        pn, img_bytes = args
        compressed = _compress_for_ai(img_bytes)
        data = extract_slide_blocks(compressed, api_key)
        return pn, data

    max_ai_workers = min(config.MAX_OCR_WORKERS, total_pages)
    args_list = [(pn, rendered_pages[pn]) for pn in range(total_pages)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_ai_workers) as ex:
        futs = {ex.submit(_extract_page, args): args[0] for args in args_list}
        for fut in concurrent.futures.as_completed(futs):
            page_num = futs[fut]
            try:
                pn, data = fut.result(timeout=120)
                extracted_pages[pn] = data
            except Exception as e:
                logger.warning(f"Slide {page_num+1} extraction failed: {e}")
                extracted_pages[page_num] = {}
            completed += 1
            if status_text:
                status_text.info(f"🔍 استخراج النصوص... ({completed}/{total_pages})")
            if progress_bar:
                progress_bar.progress(int(8 + (completed / total_pages) * 42))

    # ── Step 2b: Batch-translate all texts with Gemini ─────────────
    if status_text:
        status_text.info("🌐 ترجمة نصوص الشرائح باستخدام Gemini...")
    if progress_bar:
        progress_bar.progress(52)

    # Flatten all (page, block_index, text_en) across all pages
    all_text_refs = []
    for pn in range(total_pages):
        blocks = extracted_pages.get(pn, {}).get('blocks', [])
        for bi, block in enumerate(blocks):
            text = block.get('text_en', '') if isinstance(block, dict) else getattr(block, 'text_en', '')
            all_text_refs.append((pn, bi, text))

    texts_only = [ref[2] for ref in all_text_refs]
    translated_ar = translate_texts_gemini(texts_only, api_key)

    # Write Arabic translations back into the block dicts
    for (pn, bi, _), ar_text in zip(all_text_refs, translated_ar):
        page_data = extracted_pages.get(pn, {})
        blocks = page_data.get('blocks', [])
        if bi < len(blocks):
            if isinstance(blocks[bi], dict):
                blocks[bi]['text_ar'] = ar_text
            else:
                # Convert Pydantic object to dict and add text_ar
                blocks[bi] = {**blocks[bi].model_dump(), 'text_ar': ar_text}

    translated_pages = extracted_pages

    if progress_bar:
        progress_bar.progress(73)

    # ── Step 3: Compose output images ─────────────────────────────
    render_fn = render_arabic_only_slide if layout == 'arabic_only' else render_bilingual_slide
    label_map = {'side_by_side': 'المقارنة الثنائية', 'arabic_only': 'الترجمة العربية الكاملة'}
    if status_text:
        status_text.info(f"🖼️ بناء صور {label_map.get(layout, '')}...")
    if progress_bar:
        progress_bar.progress(75)

    slide_images = []
    for pn in range(total_pages):
        translation_data = translated_pages.get(pn, {})
        try:
            png_bytes = render_fn(rendered_pages[pn], translation_data)
            slide_images.append(png_bytes)
        except Exception as e:
            logger.warning(f"Slide {pn+1} render failed: {e}")
            slide_images.append(rendered_pages.get(pn, b''))

    if progress_bar:
        progress_bar.progress(88)

    # ── Step 4: Bundle into PDF ────────────────────────────────────
    if status_text:
        status_text.info("📄 تجميع الملف النهائي...")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        out_path = tmp.name

    if images_to_pdf([s for s in slide_images if s], out_path, fit_a4_landscape=True):
        if progress_bar:
            progress_bar.progress(100)
        return out_path

    logger.error("process_as_slides: images_to_pdf failed")
    return None
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Deprecated/Legacy wrapper (can be removed if app.py is fully updated)
def process_pdf(uploaded_file, api_key, progress_callback=None, status_callback=None):

    return process_document(uploaded_file, 'pdf', api_key, progress_callback, status_callback)

def process_document(uploaded_file, file_type, api_key, progress_callback=None, status_callback=None):
    """
    Orchestrates the whole process: Extract -> Translate -> Generate PDF & DOCX.
    Returns (pdf_path, docx_path) or (None, None) on failure.
    """
    
    # 1. Extraction (Rich Content)
    if status_callback: status_callback(f"Extracting content from {file_type.upper()}...")
    
    if hasattr(uploaded_file, 'seek'):
        uploaded_file.seek(0)
    
    # extract_content now returns a list of dictionaries (content items)
    content_items = extract_content(uploaded_file, file_type, api_key=api_key)
    logger.info(f"Extracted {len(content_items)} items.")

    if not content_items:
        logger.error("No content found in document.")
        return None, None

    # 2. Preparation for Translation
    # Filter text items that need translation
    text_items_to_translate = []
    text_indices = []
    
    for i, item in enumerate(content_items):
        if item['type'] == 'text':
            # Use the 'plain' text for translation logic (sending to API)
            text_items_to_translate.append(item['plain'])
            text_indices.append(i)
            
    total_text_items = len(text_items_to_translate)
    
    # 3. Translation Loop
    if status_callback: status_callback("Translating text...")
    
    if total_text_items > 0:
        for i, (idx_in_batch, original_text, translation) in enumerate(translate_paragraphs_concurrently(text_items_to_translate, api_key)):
            # Find the original item index
            original_item_index = text_indices[idx_in_batch]
            
            # Store translation in the item
            content_items[original_item_index]['translation'] = translation
            
            if progress_callback:
                progress_callback((i + 1) / total_text_items)
    
    # 4. Generate Documents
    if status_callback: status_callback("Generating formatted documents...")
    
    # Generate both
    pdf_path, docx_path = generate_documents(content_items)

    return pdf_path, docx_path


def _is_valid_table(table, page_w, page_h):
    """Filters out false-positive tables (such as decorative full-page borders or frames)."""
    try:
        rows = table.extract()
        if not rows or len(rows) < 2:
            return False

        tw = table.bbox[2] - table.bbox[0]
        th = table.bbox[3] - table.bbox[1]

        # If it covers a massive portion of the page (typical of whole-page frames/borders)
        if page_w > 0 and page_h > 0 and (tw * th) / (page_w * page_h) > 0.70:
            for r in rows:
                for c in r:
                    if c and len(str(c).strip()) > 250:
                        return False

        # Must have at least 2 rows with at least 2 non-empty cells
        valid_rows = 0
        for r in rows:
            if sum(1 for c in r if c and str(c).strip()) >= 2:
                valid_rows += 1
        return valid_rows >= 2
    except Exception:
        return False


# ============================================================
#  BILINGUAL OVERLAY MODE — English + Arabic on original design
# ============================================================

def _append_reflowed_overlay_page(source_doc, page_number, output_doc, events,
                                  transparent, font_path):
    """Expand one PDF page at translation points, then paginate the result.

    Original page content is copied in horizontal vector-preserving slices.
    When a translation does not fit in the existing gap, later source content
    is shifted down. The expanded page is then split into normal-sized pages.
    """
    src_page = source_doc[page_number]
    page_width, page_height = src_page.rect.width, src_page.rect.height
    # Keep the original footer (especially the page number) anchored to the
    # first output page.  Copying the whole source page through the flow slices
    # otherwise drags the footer into a continuation page when content spills.
    footer_height = min(24.0, page_height * 0.06)
    # Continuation pages need the same breathing room as a normal page.  The
    # old page-break shift landed overflow at y=0, so the first carried block
    # touched the top edge while the original page kept its source margin.
    flow_top_margin = min(30.0, max(18.0, page_height * 0.05))
    content_bottom = page_height - footer_height
    # Avoid treating the empty lower part of a page as content. Shifting that
    # whitespace along with the text can otherwise force an unnecessary page.
    source_content_bottom = 0.0
    try:
        for _kind, bbox in src_page.get_bboxlog():
            content_rect = fitz.Rect(bbox)
            if content_rect.y0 < content_bottom and content_rect.y1 > 0:
                source_content_bottom = max(
                    source_content_bottom,
                    min(content_rect.y1, content_bottom),
                )
    except Exception:
        pass
    if source_content_bottom <= 0:
        source_content_bottom = max(
            (min(block['bbox'][3], content_bottom)
             for block in src_page.get_text("dict").get("blocks", [])
             if block.get("type") == 0),
            default=0.0,
        )
    flow_font_name = "AR-Flow"
    with _FLOW_FONT_LOCK:
        if flow_font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(flow_font_name, font_path))
    bg_color = (0.93, 0.96, 1.0)
    border_color = (0.78, 0.85, 0.95)

    def _make_paragraph(text, fontsize, width):
        prepared_text = process_arabic_text(
            str(text or ""), width=max(8.0, width - 8),
            font_name=flow_font_name, font_size=fontsize, base_dir="R",
        )
        style = ParagraphStyle(
            "ArabicFlow", fontName=flow_font_name, fontSize=fontsize,
            leading=fontsize * 1.38, alignment=TA_RIGHT,
            textColor=colors.Color(*( (0.42, 0.12, 0.65) if transparent else (0.06, 0.20, 0.50) )),
            spaceBefore=0, spaceAfter=0, leftIndent=0, rightIndent=0,
        )
        paragraph = Paragraph(prepared_text, style)
        _, paragraph_height = paragraph.wrap(max(8.0, width - 8), 100000)
        return paragraph, max(12.0, paragraph_height + 4)

    def _flow_break_shift(top, bottom):
        page_index = max(0, int(top // page_height))
        page_top = page_index * page_height
        page_bottom = page_top + page_height
        usable_bottom = page_bottom - footer_height
        if top >= usable_bottom or bottom > usable_bottom:
            return max(0.0, page_bottom + flow_top_margin - top)
        return 0.0

    prepared = []
    for event in sorted(events, key=lambda item: item["source_y"]):
        rows = []
        content_height = 0.0
        for row in event.get("rows", []):
            prepared_row = []
            row_height = 0.0
            for cell in row:
                x0 = max(4.0, float(cell["x0"]))
                x1 = min(page_width - 4.0, float(cell["x1"]))
                fontsize = float(cell.get("fontsize", 8.5))
                if not str(cell.get("text", "")).strip():
                    continue
                paragraph, cell_height = _make_paragraph(
                    cell.get("text", ""), fontsize, x1 - x0,
                )
                prepared_row.append({
                    "x0": x0, "x1": x1, "fontsize": fontsize,
                    "paragraph": paragraph, "height": cell_height,
                })
                row_height = max(row_height, cell_height)
            if prepared_row:
                rows.append((prepared_row, row_height))
                content_height += row_height + 2.0

        if rows:
            y = min(page_height, max(0.0, float(event["source_y"])))
            next_y = min(page_height, max(y, float(event.get("next_y", page_height))))
            added_height = max(0.0, y + 2.0 + content_height - next_y)
            prepared.append({
                "source_y": y, "rows": rows, "height": content_height,
                "delta": added_height,
                "source_start": min(y, max(0.0, float(event.get("source_start", y)))),
            })

    # Simulate the exact shifts used below and size the expanded page from the
    # last real source/translation content. Using page_height + total_extra
    # counts blank whitespace at the bottom as content and often adds an empty
    # continuation page even when all material still fits on this page.
    simulated_shift = 0.0
    simulated_source_y = 0.0
    used_bottom = source_content_bottom
    for event in prepared:
        block_start = max(simulated_source_y, event["source_start"])
        block_top = block_start + simulated_shift
        block_bottom = event["source_y"] + simulated_shift + 2.0 + event["height"]
        added_break = _flow_break_shift(block_top, block_bottom)
        if added_break > 0:
            simulated_shift += added_break
            block_bottom += added_break
        used_bottom = max(used_bottom, block_bottom)
        simulated_shift += event["delta"]
        simulated_source_y = event["source_y"]

    used_bottom = max(used_bottom, source_content_bottom + simulated_shift)
    expanded_height = max(page_height, used_bottom)
    # Keep output pages the original size so page viewers and printers behave normally.
    expanded_height = max(page_height, int((expanded_height + page_height - 1) // page_height) * page_height)
    expanded_doc = fitz.open()
    expanded_page = expanded_doc.new_page(width=page_width, height=expanded_height)

    source_y = 0.0
    shift = 0.0
    overlays = []

    def _copy_source_slice(y0, y1, vertical_shift):
        clip_y1 = min(float(y1), content_bottom)
        clip_y0 = min(float(y0), content_bottom)
        if clip_y1 <= clip_y0:
            return
        source_clip = fitz.Rect(0, clip_y0, page_width, clip_y1)
        target_rect = fitz.Rect(
            0, clip_y0 + vertical_shift, page_width, clip_y1 + vertical_shift,
        )
        expanded_page.show_pdf_page(
            target_rect, source_doc, page_number, clip=source_clip,
        )

    for event in prepared:
        event_y = event["source_y"]
        block_start = max(source_y, event["source_start"])
        if block_start > source_y:
            _copy_source_slice(source_y, block_start, shift)

        # Keep an original paragraph together at page boundaries.  Without
        # this, slicing the expanded page can leave one or two orphan lines at
        # the top of a continuation page.
        block_top = block_start + shift
        block_bottom = event_y + shift + 2.0 + event["height"]
        shift += _flow_break_shift(block_top, block_bottom)

        if event_y > block_start:
            _copy_source_slice(block_start, event_y, shift)

        overlays.append((event, event_y + shift + 2.0))
        shift += event["delta"]
        source_y = event_y

    if source_y < source_content_bottom:
        _copy_source_slice(source_y, source_content_bottom, shift)

    # Reapply the source footer once at its original position on the first page.
    footer_clip = fitz.Rect(0, content_bottom, page_width, page_height)
    expanded_page.show_pdf_page(
        footer_clip, source_doc, page_number, clip=footer_clip, overlay=True,
    )

    from reportlab.pdfgen import canvas as reportlab_canvas
    overlay_buffer = io.BytesIO()
    overlay_canvas = reportlab_canvas.Canvas(
        overlay_buffer, pagesize=(page_width, expanded_height),
        pageCompression=1,
    )
    for event, row_y in overlays:
        for row, row_height in event["rows"]:
            for cell in row:
                x0, x1 = cell["x0"], cell["x1"]
                box_bottom = expanded_height - (row_y + row_height)
                if not transparent:
                    overlay_canvas.setFillColorRGB(*bg_color)
                    overlay_canvas.setStrokeColorRGB(*border_color)
                    overlay_canvas.rect(x0, box_bottom, x1 - x0, row_height, fill=1, stroke=1)
                paragraph = cell["paragraph"]
                _, paragraph_height = paragraph.wrap(max(8.0, x1 - x0 - 8), row_height - 2)
                paragraph.drawOn(
                    overlay_canvas, x0 + 4,
                    box_bottom + max(1.0, (row_height - paragraph_height) / 2),
                )
    overlay_canvas.save()
    overlay_buffer.seek(0)
    overlay_doc = fitz.open(stream=overlay_buffer.getvalue(), filetype="pdf")
    expanded_page.show_pdf_page(expanded_page.rect, overlay_doc, 0, overlay=True)
    overlay_doc.close()

    page_count = max(1, int(expanded_height // page_height))
    for page_index in range(page_count):
        output_page = output_doc.new_page(width=page_width, height=page_height)
        y0 = page_index * page_height
        output_page.show_pdf_page(
            fitz.Rect(0, 0, page_width, page_height), expanded_doc, 0,
            clip=fitz.Rect(0, y0, page_width, y0 + page_height),
        )

    expanded_doc.close()
    return page_count

def generate_bilingual_overlay_pdf(pdf_path, api_key, output_path,
                                    status_text=None, progress_bar=None,
                                    transparent=False):
    """
    Creates a bilingual PDF: keeps the original English text and inserts
    Arabic translation below each text block. Transparent mode omits the
    translation backgrounds and uses red Arabic text over the original page.
    The page is extended vertically to accommodate the added translations.
    For scanned PDFs, it uses OCR via Gemini Vision.
    Returns True on success.
    """
    from translator import (translate_paragraphs_concurrently,
                            generate_document_summary, translate_table,
                            extract_and_translate_page)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    font_path = os.path.abspath(config.ARABIC_FONT_FILE)

    if not os.path.exists(font_path):
        download_arabic_font()

    scanned = is_scanned_pdf(pdf_path)
    if status_text:
        mode_label = "مصوّر (OCR)" if scanned else "نصي"
        status_text.info(f"📖 نوع الملف: {mode_label} — تحليل الصفحات...")

    # ═══════════════════════════════════════════════════════════
    #  SCANNED PATH — use Gemini Vision OCR per page
    # ═══════════════════════════════════════════════════════════
    if scanned:
        import concurrent.futures
        from PIL import ImageEnhance, ImageFilter

        out_doc = fitz.open()  # new empty output document

        def _ocr_page(page_num):
            page = doc[page_num]
            dpi = _adaptive_dpi(page)
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("jpeg")
            # Enhance
            img = PILImage.open(io.BytesIO(img_bytes))
            img = img.filter(ImageFilter.MedianFilter(3))
            img = ImageEnhance.Sharpness(img).enhance(1.4)
            img = ImageEnhance.Contrast(img).enhance(1.3)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            enhanced = _compress_for_api(buf.getvalue(), max_size_kb=500)
            items = extract_and_translate_page(enhanced, api_key)
            return page_num, items

        if status_text:
            status_text.info(f"🔍 قراءة وترجمة {total_pages} صفحة بالذكاء الاصطناعي...")

        max_w = min(config.MAX_OCR_WORKERS, total_pages)
        results_map = {}
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
            futs = {executor.submit(_ocr_page, pn): pn for pn in range(total_pages)}
            for fut in concurrent.futures.as_completed(futs):
                pn = futs[fut]
                try:
                    pn, items = fut.result(timeout=120)
                    results_map[pn] = items
                except Exception as e:
                    logger.error(f"OCR page {pn+1} failed: {e}")
                    results_map[pn] = []
                completed += 1
                if progress_bar:
                    progress_bar.progress(int((completed / total_pages) * 60))

        # ── Build output pages ──
        AR_BG = (0.93, 0.96, 1.0)       # soft blue tint
        AR_BORDER = (0.78, 0.85, 0.95)
        AR_TEXT_COLOR = (0.42, 0.12, 0.65) if transparent else (0.06, 0.20, 0.50)
        SEPARATOR_COLOR = (0.80, 0.85, 0.92)

        for pn in range(total_pages):
            src_page = doc[pn]
            pw, ph = src_page.rect.width, src_page.rect.height
            items = results_map.get(pn, [])

            # Filter to text items with translations
            text_items = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                if it.get('type') != 'text':
                    continue
                ar = it.get('arabic', '').strip()
                if ar:
                    text_items.append(it)

            if transparent and text_items:
                ocr_blocks = []
                for it in text_items:
                    if 'x_pct' in it and 'y_pct' in it:
                        x0 = (it['x_pct'] / 100.0) * pw
                        x1 = x0 + (it.get('w_pct', 80) / 100.0) * pw
                        y0 = (it['y_pct'] / 100.0) * ph
                        y1 = ((it['y_pct'] + it.get('h_pct', 5)) / 100.0) * ph
                    else:
                        x0, x1 = pw * 0.05, pw * 0.95
                        y0, y1 = ph * 0.78, ph * 0.84
                    ocr_blocks.append((it, x0, min(pw - 5, x1), y0, min(ph, y1)))

                multi_column = any(
                    max(a[3], b[3]) < min(a[4], b[4]) and
                    (a[2] <= b[1] or b[2] <= a[1])
                    for i, a in enumerate(ocr_blocks) for b in ocr_blocks[i + 1:]
                )
                if not multi_column:
                    flow_events = []
                    for block_index, (it, x0, x1, y0, y1) in enumerate(ocr_blocks):
                        next_y = min(
                            [other[3] for j, other in enumerate(ocr_blocks)
                             if j != block_index and other[3] >= y1 and
                             max(x0, other[1]) < min(x1, other[2])] + [ph]
                        )
                        flow_events.append({
                            "source_y": y1,
                            "source_start": y0,
                            "next_y": next_y,
                            "rows": [[{
                                "x0": x0, "x1": x1,
                                "text": it.get('arabic', '').strip(),
                                "fontsize": 8.5,
                            }]],
                        })
                    _append_reflowed_overlay_page(
                        doc, pn, out_doc, flow_events, transparent=True,
                        font_path=font_path,
                    )
                    if progress_bar:
                        progress_bar.progress(int(60 + ((pn + 1) / total_pages) * 35))
                    continue

            # Maintain exact original page dimensions without dynamic height expansion
            new_page = out_doc.new_page(width=pw, height=ph)
            new_page.show_pdf_page(fitz.Rect(0, 0, pw, ph), doc, pn)

            ar_font_size = 8.5
            line_height = 12

            for it in text_items:
                ar_raw = it.get('arabic', '')
                if not ar_raw.strip():
                    continue

                # Reshape Arabic
                ar_clean = normalize_arabic_text(ar_raw)
                ar_reshaped = arabic_reshaper.reshape(ar_clean)
                ar_display = get_display(ar_reshaped, base_dir='R')

                # If position coordinates from vision model are present
                if 'x_pct' in it and 'y_pct' in it:
                    bx0 = (it['x_pct'] / 100.0) * pw
                    by1 = ((it['y_pct'] + it.get('h_pct', 5)) / 100.0) * ph
                    bw = (it.get('w_pct', 80) / 100.0) * pw
                    bw = max(bw, 40)
                    bh = 16
                    box_rect = fitz.Rect(bx0, by1 + 2, min(pw - 5, bx0 + bw), min(ph - 5, by1 + 2 + bh))
                else:
                    margin_x = pw * 0.05
                    box_rect = fitz.Rect(margin_x, min(ph - 25, ph * 0.85), pw - margin_x, min(ph - 5, ph * 0.85 + 20))

                if not transparent:
                    shape = new_page.new_shape()
                    shape.draw_rect(box_rect)
                    shape.finish(fill=AR_BG, color=AR_BORDER, width=0.5, fill_opacity=0.9)
                    shape.commit()

                text_r = fitz.Rect(box_rect.x0 + 4, box_rect.y0 + 2, box_rect.x1 - 4, box_rect.y1 - 2)
                try:
                    new_page.insert_textbox(
                        text_r, ar_display,
                        fontname="ar-bilingual", fontfile=font_path,
                        fontsize=ar_font_size,
                        align=fitz.TEXT_ALIGN_RIGHT,
                        color=AR_TEXT_COLOR
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert bilingual textbox (scanned): {e}")

            if progress_bar:
                progress_bar.progress(int(60 + ((pn + 1) / total_pages) * 35))

        out_doc.save(output_path, garbage=4, deflate=True)
        out_doc.close()
        doc.close()
        if progress_bar:
            progress_bar.progress(100)
        return True

    # ═══════════════════════════════════════════════════════════
    #  TEXT-BASED PATH — extract text blocks from native PDF
    # ═══════════════════════════════════════════════════════════

    # ── Phase 1: Detect repetitive headers/footers ────────────
    block_freq = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            text = " ".join(s["text"] for ln in block["lines"] for s in ln["spans"]).strip()
            text = re.sub(r'\s+', ' ', text)
            if not text or len(text) >= 200:
                continue
            yc = (block["bbox"][1] + block["bbox"][3]) / 2
            ph = page.rect.height
            if ph > 0 and (yc / ph < 0.12 or yc / ph > 0.88):
                block_freq[text] = block_freq.get(text, 0) + 1

    rep_thr = max(2, int(total_pages * 0.3)) if total_pages > 3 else 999
    repetitive = {t for t, c in block_freq.items() if c >= rep_thr}
    page_num_pats = [re.compile(r'^\d+$'),
                     re.compile(r'^(page|pg\.?)\s*\|?\s*\d+$', re.I),
                     re.compile(r'^\d+\s*[\/\-]\s*\d+$')]

    # ── Phase 2: Collect translatable text per page ───────────
    page_text_blocks = {}   # page -> [{bbox, text, color, size, bold}]
    page_table_items = {}   # page -> [{bbox, rows, cells_bboxes, col_count}]

    all_paragraphs = []
    para_map = []           # (page, block_idx)
    tables_to_translate = []

    for pn in range(total_pages):
        page = doc[pn]
        t_blocks, t_tables = [], []

        # Tables first
        tables = page.find_tables()
        table_rects = []
        if tables:
            for table in tables:
                if not _is_valid_table(table, page.rect.width, page.rect.height):
                    continue
                table_rects.append(fitz.Rect(table.bbox))
                raw_rows = []
                for row in table.extract():
                    raw_rows.append([str(c).replace("\n", " ").strip() if c else "" for c in row])
                col_count = table.col_count if hasattr(table, 'col_count') else (len(raw_rows[0]) if raw_rows else 0)
                cells_bb = list(table.cells) if hasattr(table, 'cells') else []
                t_tables.append({
                    'bbox': table.bbox, 'rows': raw_rows,
                    'cells_bboxes': cells_bb, 'col_count': col_count,
                })
                tables_to_translate.append((pn, len(t_tables) - 1, raw_rows))

        # Text blocks
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            br = fitz.Rect(block["bbox"])
            center = br.tl + (br.br - br.tl) * 0.5
            if any(tr.contains(center) for tr in table_rects):
                continue
            text = re.sub(r'\s+', ' ',
                          " ".join(s["text"] for ln in block["lines"] for s in ln["spans"])).strip()
            if not text:
                continue
            if len(text) < 20 and any(p.match(text) for p in page_num_pats):
                continue
            if text in repetitive:
                continue
            first_span = block["lines"][0]["spans"][0]
            color = first_span.get("color", 0)
            size = first_span.get("size", 11)
            is_bold = (first_span.get("flags", 0) & 16) > 0 or "Bold" in first_span.get("font", "")

            t_blocks.append({
                'bbox': block["bbox"], 'text': text,
                'color': color, 'size': size, 'bold': is_bold
            })
            para_map.append((pn, len(t_blocks) - 1))
            all_paragraphs.append(text)

        page_text_blocks[pn] = t_blocks
        page_table_items[pn] = t_tables

    if progress_bar:
        progress_bar.progress(15)

    # ── Phase 3: Translate everything ─────────────────────────
    if status_text:
        status_text.info(f"🌐 ترجمة {len(all_paragraphs)} فقرة و {len(tables_to_translate)} جدول...")

    if status_text:
        status_text.info("🔎 تحليل سياق المحاضرة قبل بدء الترجمة...")
    sample = " ".join(all_paragraphs[:50])
    context = generate_document_summary(sample, api_key) if config.USE_DOCUMENT_SUMMARY and sample.strip() else ""
    if status_text:
        batch_count = get_translation_batch_count(all_paragraphs)
        status_text.info(f"🌐 بدأت الترجمة: {len(all_paragraphs)} فقرة في {batch_count} دفعات...")

    translations = [""] * len(all_paragraphs)
    if all_paragraphs:
        completed = 0
        for idx, _orig, trans in translate_paragraphs_concurrently(all_paragraphs, api_key, context):
            translations[idx] = trans
            completed += 1
            if status_text and (completed % 15 == 0 or completed == len(all_paragraphs)):
                status_text.info(f"🌐 اكتملت ترجمة {completed}/{len(all_paragraphs)} فقرة...")
            if progress_bar:
                progress_bar.progress(int(15 + (completed / len(all_paragraphs)) * 50))

    for i, (pn, bi) in enumerate(para_map):
        page_text_blocks[pn][bi]['translation'] = translations[i]

    # Translate tables
    import concurrent.futures
    if tables_to_translate:
        def _tr_table(args):
            pn, ti, rows = args
            return pn, ti, translate_table(rows, api_key, context)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(config.MAX_WORKERS, len(tables_to_translate))) as ex:
            for fut in concurrent.futures.as_completed(
                    {ex.submit(_tr_table, t): t for t in tables_to_translate}):
                try:
                    pn, ti, tr_rows = fut.result()
                    page_table_items[pn][ti]['translated_rows'] = tr_rows
                except GeminiQuotaExceeded:
                    raise
                except Exception as e:
                    logger.error(f"Table translation error: {e}")

    if progress_bar:
        progress_bar.progress(70)

    # ── Phase 4: Build bilingual pages ────────────────────────
    if status_text:
        status_text.info("🎨 بناء الملف الثنائي (إنجليزي + عربي)...")

    AR_BG = (0.93, 0.96, 1.0)           # soft blue tint
    AR_BORDER = (0.78, 0.85, 0.95)
    AR_TEXT_COLOR = (0.42, 0.12, 0.65) if transparent else (0.06, 0.20, 0.50)

    def _wrap_arabic_multiline(text, max_w, font, fontsize):
        words = text.split()
        if not words:
            return "", 1
        lines = []
        curr_line = []
        for word in words:
            test_line = ' '.join(curr_line + [word])
            reshaped = arabic_reshaper.reshape(test_line)
            if font.text_length(reshaped, fontsize=fontsize) <= max_w:
                curr_line.append(word)
            else:
                if curr_line:
                    lines.append(' '.join(curr_line))
                curr_line = [word]
        if curr_line:
            lines.append(' '.join(curr_line))

        # BiDi applied to each wrapped line individually so vertical order is always top-to-bottom
        bidi_lines = [get_display(arabic_reshaper.reshape(ln), base_dir='R') for ln in lines]
        return '\n'.join(bidi_lines), max(1, len(lines))

    fitz_font = fitz.Font(fontfile=font_path) if os.path.exists(font_path) else fitz.Font("helv")
    out_doc = fitz.open()  # new output document

    def _aligned_column_right(block, candidates, page_width):
        """Use the right edge of neighboring lines in the same text column.

        PDFs often expose a short last line (for example, just "change.") as
        its own block. Using only that line's narrow bbox forces its Arabic
        translation to break one word per line. Lines sharing a left edge are
        a reliable lightweight signal for the same column.
        """
        bx0 = block['bbox'][0]
        tolerance = max(12.0, float(block.get('size', 11)) * 1.5)
        right_edges = [
            other['bbox'][2]
            for other in candidates
            if abs(other['bbox'][0] - bx0) <= tolerance
        ]
        return min(page_width - 12.0, max([block['bbox'][2], *right_edges]))

    for pn in range(total_pages):
        src_page = doc[pn]
        pw, ph = src_page.rect.width, src_page.rect.height

        blocks = [b for b in page_text_blocks.get(pn, []) if b.get('translation', '').strip()]
        blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))

        # Detect multi-column layout (blocks horizontally separated while overlapping vertically).
        layout_blocks = page_text_blocks.get(pn, [])
        is_multi_column = False
        for i, b1 in enumerate(layout_blocks):
            for b2 in layout_blocks[i+1:]:
                v_overlap = max(b1['bbox'][1], b2['bbox'][1]) < min(b1['bbox'][3], b2['bbox'][3])
                h_separate = (b1['bbox'][2] < b2['bbox'][0]) or (b2['bbox'][2] < b1['bbox'][0])
                if v_overlap and h_separate:
                    is_multi_column = True
                    break
            if is_multi_column:
                break

        if transparent and not is_multi_column:
            flow_events = []
            page_tables = page_table_items.get(pn, [])

            for block in blocks:
                bx0, by0, bx1, by1 = block['bbox']
                x0 = max(8.0, bx0)
                inferred_right = _aligned_column_right(block, layout_blocks, pw)
                x1 = min(pw - 8.0, max(x0 + 30.0, bx1, inferred_right))
                next_positions = [
                    other['bbox'][1] for other in layout_blocks
                    if other is not block and other['bbox'][1] >= by1 - 0.5
                    and max(x0, other['bbox'][0]) < min(x1, other['bbox'][2])
                ]
                next_positions.extend(
                    fitz.Rect(table['bbox']).y0 for table in page_tables
                    if fitz.Rect(table['bbox']).y0 >= by1 - 0.5
                    and max(x0, fitz.Rect(table['bbox']).x0) < min(x1, fitz.Rect(table['bbox']).x1)
                )
                next_y = min(next_positions + [ph - 12.0])
                flow_events.append({
                    "source_y": by1,
                    "source_start": by0,
                    "next_y": next_y,
                    "rows": [[{
                        "x0": x0, "x1": x1,
                        "text": block.get('translation', '').strip(),
                        # Keep the translation readable beside the source text.
                        # The previous 0.68 scale made it visibly smaller than
                        # the reference bot and difficult to read on phones.
                        "fontsize": min(12.5, max(8.5, block.get('size', 11) * 0.85)),
                    }]],
                })

            for table in page_tables:
                translated_rows = table.get('translated_rows') or []
                if not translated_rows:
                    continue
                table_rect = fitz.Rect(table['bbox'])
                col_count = table.get('col_count', 0)
                cells = table.get('cells_bboxes', [])
                table_rows = []
                for row_index, translated_row in enumerate(translated_rows):
                    cell_items = []
                    for col_index, text in enumerate(translated_row):
                        cell_index = row_index * col_count + col_index
                        if cell_index >= len(cells) or cells[cell_index] is None or not str(text).strip():
                            continue
                        cell_rect = fitz.Rect(cells[cell_index])
                        cell_items.append({
                            "x0": cell_rect.x0 + 1,
                            "x1": cell_rect.x1 - 1,
                            "text": str(text).strip(),
                            "fontsize": 6.8,
                        })
                    if cell_items:
                        table_rows.append(cell_items)
                if table_rows:
                    next_positions = [
                        block['bbox'][1] for block in layout_blocks
                        if block['bbox'][1] >= table_rect.y1 - 0.5
                        and max(table_rect.x0, block['bbox'][0]) < min(table_rect.x1, block['bbox'][2])
                    ]
                    next_positions.extend(
                        fitz.Rect(other['bbox']).y0 for other in page_tables
                        if other is not table and fitz.Rect(other['bbox']).y0 >= table_rect.y1 - 0.5
                        and max(table_rect.x0, fitz.Rect(other['bbox']).x0) < min(table_rect.x1, fitz.Rect(other['bbox']).x1)
                    )
                    flow_events.append({
                        "source_y": table_rect.y1,
                        "source_start": table_rect.y0,
                        "next_y": min(next_positions + [ph - 12.0]),
                        "rows": table_rows,
                    })

            if flow_events:
                _append_reflowed_overlay_page(
                    doc, pn, out_doc, flow_events, transparent=True,
                    font_path=font_path,
                )
                if progress_bar:
                    progress_bar.progress(int(70 + ((pn + 1) / total_pages) * 25))
                continue

        # Maintain exact original page dimensions without dynamic height expansion or slicing
        new_page = out_doc.new_page(width=pw, height=ph)
        new_page.show_pdf_page(fitz.Rect(0, 0, pw, ph), doc, pn)

        # 1. Render text block translations
        for b in blocks:
            text = b.get('text', '').strip()
            trans = b.get('translation', '').strip()
            if not text or not trans:
                continue

            bx0, by0, bx1, by1 = b['bbox']
            bw = bx1 - bx0

            # Column boundary constraint: ensure box stays inside its column/card
            col_right = _aligned_column_right(b, layout_blocks, pw)
            for other in blocks:
                if other is b:
                    continue
                if other['bbox'][0] >= bx0 + 20 and max(by0 - 20, other['bbox'][1]) < min(by1 + 100, other['bbox'][3]):
                    col_right = min(col_right, other['bbox'][0] - 8)

            avail_w = max(bw, col_right - bx0)
            avail_w = max(avail_w, 40)
            avail_w = min(avail_w, pw - bx0 - 15)

            # Vertical space available before next element in same column
            next_y = ph - 15
            for other in blocks:
                if other is b:
                    continue
                h_overlap = max(bx0, other['bbox'][0]) < min(bx0 + avail_w, other['bbox'][2])
                if h_overlap and other['bbox'][1] >= by1:
                    next_y = min(next_y, other['bbox'][1])

            for tbl in page_table_items.get(pn, []):
                t_rect = fitz.Rect(tbl['bbox'])
                if max(bx0, t_rect.x0) < min(bx0 + avail_w, t_rect.x1) and t_rect.y0 >= by1:
                    next_y = min(next_y, t_rect.y0)

            avail_h = max(0, next_y - by1 - 4)

            fontsize = 8.5
            wrapped_ar, num_lines = _wrap_arabic_multiline(trans, avail_w - 12, fitz_font, fontsize)
            bh = num_lines * 13 + 6

            if bh > avail_h and avail_h >= 14:
                fontsize = 7.0
                wrapped_ar, num_lines = _wrap_arabic_multiline(trans, avail_w - 12, fitz_font, fontsize)
                bh = min(avail_h, num_lines * 10 + 4)
            elif bh > avail_h and avail_h < 14:
                fontsize = 6.5
                wrapped_ar, num_lines = _wrap_arabic_multiline(trans, avail_w - 10, fitz_font, fontsize)
                bh = max(10, min(14, avail_h if avail_h > 8 else 12))

            box_rect = fitz.Rect(bx0, by1 + 2, bx0 + avail_w, by1 + 2 + bh)
            box_rect = box_rect & fitz.Rect(4, 4, pw - 4, ph - 4)

            if not transparent:
                shape = new_page.new_shape()
                shape.draw_rect(box_rect)
                shape.finish(fill=AR_BG, color=AR_BORDER, width=0.6)
                shape.commit()

            text_rect = fitz.Rect(box_rect.x0 + 4, box_rect.y0 + 2, box_rect.x1 - 4, box_rect.y1 - 2)
            try:
                new_page.insert_textbox(
                    text_rect, wrapped_ar,
                    fontname="ar-bilingual", fontfile=font_path,
                    fontsize=fontsize,
                    align=fitz.TEXT_ALIGN_RIGHT,
                    color=AR_TEXT_COLOR
                )
            except Exception as e:
                logger.warning(f"Failed to insert smart bilingual textbox: {e}")

        # 2. Render table cell translations
        for tbl in page_table_items.get(pn, []):
            tr_rows = tbl.get('translated_rows')
            if not tr_rows:
                continue
            cells_bb = tbl.get('cells_bboxes', [])
            col_count = tbl.get('col_count', 0)
            if not cells_bb or col_count == 0:
                continue

            for r_idx, row in enumerate(tr_rows):
                for c_idx, cell_text in enumerate(row):
                    if not cell_text or not str(cell_text).strip():
                        continue
                    cell_idx = r_idx * col_count + c_idx
                    if cell_idx >= len(cells_bb):
                        continue

                    cb = cells_bb[cell_idx]
                    if cb is None:
                        continue
                    cell_rect = fitz.Rect(cb)
                    if cell_rect.is_empty or cell_rect.width < 15 or cell_rect.height < 10:
                        continue

                    cell_h = cell_rect.height
                    ar_h = min(cell_h * 0.48, max(13, cell_h - 10))
                    ar_h = min(ar_h, 30)
                    ar_fs = min(7.5, max(5.0, ar_h * 0.45))

                    ar_cell_rect = fitz.Rect(
                        cell_rect.x0 + 1.5,
                        cell_rect.y1 - ar_h - 1.5,
                        cell_rect.x1 - 1.5,
                        cell_rect.y1 - 1.5
                    )
                    ar_cell_rect = ar_cell_rect & fitz.Rect(0, 0, pw, ph)
                    if ar_cell_rect.is_empty or ar_cell_rect.height < 6:
                        continue

                    wrapped_cell_ar, num_lines = _wrap_arabic_multiline(
                        str(cell_text).strip(), ar_cell_rect.width - 4, fitz_font, ar_fs
                    )

                    if not transparent:
                        shape = new_page.new_shape()
                        shape.draw_rect(ar_cell_rect)
                        shape.finish(fill=AR_BG, color=AR_BORDER, width=0.5)
                        shape.commit()

                    try:
                        text_r = fitz.Rect(ar_cell_rect.x0 + 2, ar_cell_rect.y0 + 1, ar_cell_rect.x1 - 2, ar_cell_rect.y1 - 1)
                        new_page.insert_textbox(
                            text_r, wrapped_cell_ar,
                            fontname="ar-bilingual", fontfile=font_path,
                            fontsize=ar_fs, color=AR_TEXT_COLOR,
                            align=fitz.TEXT_ALIGN_RIGHT
                        )
                    except Exception as e:
                        logger.warning(f"Error inserting table cell text: {e}")

        if progress_bar:
            progress_bar.progress(int(70 + ((pn + 1) / total_pages) * 25))

    out_doc.save(output_path, garbage=4, deflate=True)
    out_doc.close()
    doc.close()

    if progress_bar:
        progress_bar.progress(100)
    return True


# ============================================================
#  OVERLAY MODE — Preserves original PDF layout
# ============================================================

def generate_overlay_pdf(pdf_path, api_key, output_path,
                         status_text=None, progress_bar=None):
    """
    Overlays Arabic translations on the **original** PDF pages.
    Preserves 100% of the original formatting, images, tables, and layout.
    Returns True on success.
    """
    from translator import translate_paragraphs_concurrently, generate_document_summary, translate_table

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    font_path = os.path.abspath(config.ARABIC_FONT_FILE)

    if not os.path.exists(font_path):
        download_arabic_font()

    # ── Phase 1: Detect repetitive headers/footers ────────────────
    if status_text:
        status_text.info("📖 تحليل الملف الأصلي...")

    block_freq = {}
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            text = " ".join(s["text"] for ln in block["lines"] for s in ln["spans"]).strip()
            text = re.sub(r'\s+', ' ', text)
            if not text or len(text) >= 200:
                continue
            yc = (block["bbox"][1] + block["bbox"][3]) / 2
            ph = page.rect.height
            if ph > 0 and (yc / ph < 0.12 or yc / ph > 0.88):
                block_freq[text] = block_freq.get(text, 0) + 1

    rep_thr = max(2, int(total_pages * 0.3)) if total_pages > 3 else 999
    repetitive = {t for t, c in block_freq.items() if c >= rep_thr}
    page_num_pats = [re.compile(r'^\d+$'),
                     re.compile(r'^(page|pg\.?)\s*\|?\s*\d+$', re.I),
                     re.compile(r'^\d+\s*[\/\-]\s*\d+$')]

    # ── Phase 2: Collect translatable content per page ────────────
    page_text_blocks = {}   # page -> [{bbox, text}]
    page_table_items = {}   # page -> [{bbox, rows, cells_bboxes, col_count}]

    all_paragraphs = []     # flat for batch translation
    para_map = []           # (page, block_idx)

    tables_to_translate = []  # (page, tbl_idx, rows)

    for pn in range(total_pages):
        page = doc[pn]
        t_blocks, t_tables = [], []

        # Tables first
        tables = page.find_tables()
        table_rects = []
        if tables:
            for table in tables:
                if not _is_valid_table(table, page.rect.width, page.rect.height):
                    continue
                table_rects.append(fitz.Rect(table.bbox))
                raw_rows = []
                for row in table.extract():
                    raw_rows.append([str(c).replace("\n", " ").strip() if c else "" for c in row])
                col_count = table.col_count if hasattr(table, 'col_count') else (len(raw_rows[0]) if raw_rows else 0)
                cells_bb = list(table.cells) if hasattr(table, 'cells') else []
                t_tables.append({
                    'bbox': table.bbox, 'rows': raw_rows,
                    'cells_bboxes': cells_bb, 'col_count': col_count,
                })
                tables_to_translate.append((pn, len(t_tables) - 1, raw_rows))

        # Text blocks (skip those inside tables)
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            br = fitz.Rect(block["bbox"])
            center = br.tl + (br.br - br.tl) * 0.5
            if any(tr.contains(center) for tr in table_rects):
                continue
            text = re.sub(r'\s+', ' ',
                          " ".join(s["text"] for ln in block["lines"] for s in ln["spans"])).strip()
            if not text:
                continue
            if len(text) < 20 and any(p.match(text) for p in page_num_pats):
                continue
            if text in repetitive:
                continue
            first_span = block["lines"][0]["spans"][0]
            color = first_span.get("color", 0)
            size = first_span.get("size", 11)
            is_bold = (first_span.get("flags", 0) & 16) > 0 or "Bold" in first_span.get("font", "")

            t_blocks.append({
                'bbox': block["bbox"], 
                'text': text, 
                'color': color, 
                'size': size, 
                'bold': is_bold
            })
            para_map.append((pn, len(t_blocks) - 1))
            all_paragraphs.append(text)

        page_text_blocks[pn] = t_blocks
        page_table_items[pn] = t_tables

    if progress_bar:
        progress_bar.progress(15)

    # ── Phase 3: Translate everything ─────────────────────────────
    if status_text:
        status_text.info(f"🌐 ترجمة {len(all_paragraphs)} فقرة و {len(tables_to_translate)} جدول...")

    # Context summary
    if status_text:
        status_text.info("🔎 تحليل سياق المحاضرة قبل بدء الترجمة...")
    sample = " ".join(all_paragraphs[:50])
    context = generate_document_summary(sample, api_key) if config.USE_DOCUMENT_SUMMARY and sample.strip() else ""
    if status_text:
        batch_count = get_translation_batch_count(all_paragraphs)
        status_text.info(f"🌐 بدأت الترجمة: {len(all_paragraphs)} فقرة في {batch_count} دفعات...")

    # Batch-translate paragraphs
    translations = [""] * len(all_paragraphs)
    if all_paragraphs:
        completed = 0
        for idx, _orig, trans in translate_paragraphs_concurrently(all_paragraphs, api_key, context):
            translations[idx] = trans
            completed += 1
            if status_text and (completed % 15 == 0 or completed == len(all_paragraphs)):
                status_text.info(f"🌐 اكتملت ترجمة {completed}/{len(all_paragraphs)} فقرة...")
            if progress_bar:
                progress_bar.progress(int(15 + (completed / len(all_paragraphs)) * 55))

    # Map translations back
    for i, (pn, bi) in enumerate(para_map):
        page_text_blocks[pn][bi]['translation'] = translations[i]

    # Translate tables
    import concurrent.futures
    if tables_to_translate:
        def _tr_table(args):
            pn, ti, rows = args
            return pn, ti, translate_table(rows, api_key, context)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(config.MAX_WORKERS, len(tables_to_translate))) as ex:
            for fut in concurrent.futures.as_completed(
                    {ex.submit(_tr_table, t): t for t in tables_to_translate}):
                try:
                    pn, ti, tr_rows = fut.result()
                    page_table_items[pn][ti]['translated_rows'] = tr_rows
                except GeminiQuotaExceeded:
                    raise
                except Exception as e:
                    logger.error(f"Table translation error: {e}")

    if progress_bar:
        progress_bar.progress(75)

    # ── Phase 4: Overlay translations on each page ────────────────
    if status_text:
        status_text.info("🎨 تطبيق الترجمة على الملف الأصلي...")

    BG_COLOR = (1.0, 0.93, 0.93)     # light pink
    BORDER_COLOR = (0.88, 0.78, 0.78)
    TEXT_COLOR = (0.42, 0.12, 0.65) # purple

    for pn in range(total_pages):
        page = doc[pn]
        pw, ph = page.rect.width, page.rect.height

        # ── Overlay text blocks ──
        blocks = [b for b in page_text_blocks.get(pn, []) if b.get('translation', '').strip()]
        blocks.sort(key=lambda b: b['bbox'][1])

        for i, blk in enumerate(blocks):
            bbox = blk['bbox']
            ar_raw = blk['translation']

            # Reshape Arabic
            ar_clean = normalize_arabic_text(ar_raw)
            ar_reshaped = arabic_reshaper.reshape(ar_clean)
            ar_display = get_display(ar_reshaped, base_dir='R')

            # 1. Mask original text (Draw white rectangle over the English text)
            pad = 1.0
            mask_rect = fitz.Rect(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
            page.draw_rect(mask_rect, color=None, fill=(1, 1, 1))

            # 2. Extract original styling
            c_int = blk.get('color', 0)
            r = ((c_int >> 16) & 255) / 255.0
            g = ((c_int >> 8) & 255) / 255.0
            b = (c_int & 255) / 255.0
            text_color = (r, g, b)
            
            font_size = blk.get('size', 11)

            # 3. Insert translated text in the same bounding box
            try:
                page.insert_textbox(
                    bbox, 
                    ar_display, 
                    fontname="ar-overlay", 
                    fontfile=font_path,
                    fontsize=font_size + 1,  # slightly larger to match Arabic proportions
                    align=fitz.TEXT_ALIGN_RIGHT, 
                    color=text_color
                )
            except Exception as e:
                logger.warning(f"Failed to insert overlay textbox: {e}")

        # ── Overlay table translations ──
        for tbl in page_table_items.get(pn, []):
            tr_rows = tbl.get('translated_rows')
            if not tr_rows:
                continue
            cells_bb = tbl.get('cells_bboxes', [])
            col_count = tbl.get('col_count', 0)
            if not cells_bb or col_count == 0:
                continue

            for r_idx, row in enumerate(tr_rows):
                for c_idx, cell_text in enumerate(row):
                    if not cell_text.strip():
                        continue
                    cell_idx = r_idx * col_count + c_idx
                    if cell_idx >= len(cells_bb):
                        continue

                    cb = cells_bb[cell_idx]
                    if cb is None:
                        continue
                    cell_rect = fitz.Rect(cb)
                    if cell_rect.is_empty or cell_rect.width < 15:
                        continue

                    # Place translation in bottom half of cell
                    cell_h = cell_rect.height
                    ar_h = min(cell_h * 0.45, 20)
                    ar_fs = min(6.5, max(5, ar_h * 0.6))

                    ar_cell_rect = fitz.Rect(
                        cell_rect.x0 + 1, cell_rect.y1 - ar_h - 1,
                        cell_rect.x1 - 1, cell_rect.y1 - 1
                    )
                    ar_cell_rect = ar_cell_rect & page.rect
                    if ar_cell_rect.is_empty or ar_cell_rect.height < 5:
                        continue

                    ar_text = normalize_arabic_text(cell_text)
                    ar_text = get_display(arabic_reshaper.reshape(ar_text), base_dir='R')

                    # Subtle background
                    shape = page.new_shape()
                    shape.draw_rect(ar_cell_rect)
                    shape.finish(fill=BG_COLOR, fill_opacity=0.6, color=None)
                    shape.commit()

                    try:
                        page.insert_textbox(
                            ar_cell_rect, ar_text,
                            fontname="ar-overlay", fontfile=font_path,
                            fontsize=ar_fs, color=TEXT_COLOR,
                            align=fitz.TEXT_ALIGN_RIGHT
                        )
                    except Exception:
                        pass

    if progress_bar:
        progress_bar.progress(95)

    # ── Save ──────────────────────────────────────────────────────
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    if progress_bar:
        progress_bar.progress(100)
    return True
