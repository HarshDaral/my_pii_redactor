"""
redactor.py - PDF redaction engine using spaCy NER + regex fallback.
Uses dual-model approach:
  - Custom PII model: detects EMAIL, URL_PERSONAL, ID_NUM, PHONE_NUM, NAME_STUDENT
  - en_core_web_sm: detects PERSON entities (names) in any document style
Supports both text-based and scanned (image) PDFs via optional OCR.
"""
import re
import io
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import spacy

# ── OCR support (optional) ────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Academic / public domain allowlist ───────────────────────────────────────
# Applied to both https:// and www. URLs consistently.
_EXCLUDE_URL_DOMAINS = re.compile(
    r"(?:https?://|www\.)"
    r"(?:doi\.org|dx\.doi\.org|arxiv\.org|pubmed\.ncbi|scholar\.google|"
    r"www\.ncbi\.nlm|ncbi\.nlm\.nih|orcid\.org|creativecommons\.org|"
    r"ieee\.org|springer\.com|elsevier\.com|wiley\.com|acm\.org|"
    r"researchgate\.net|academia\.edu|github\.com|"
    r"linkedin\.com/company|twitter\.com|youtube\.com|wikipedia\.org|"
    r"[a-z0-9\-]+\.gov\.in|[a-z0-9\-]+\.gov|[a-z0-9\-]+\.nic\.in|"
    r"[a-z0-9\-]+\.mil|[a-z0-9\-]+\.edu|[a-z0-9\-]+\.ac\.in|"
    r"[a-z0-9\-]+\.res\.in|[a-z0-9\-]+\.org\.in)",
    re.IGNORECASE,
)

# ── Regex patterns ────────────────────────────────────────────────────────────
REGEX_PATTERNS = {
    # Email: require a valid TLD (2-6 alpha chars, no digits/dots)
    "EMAIL": re.compile(
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}\b"
    ),

    # Phone numbers — requires word boundaries to avoid matching inside larger numbers
    "PHONE_NUM": re.compile(
        r"(?<!\d)"                                         # no digit immediately before
        r"(?:"
        # Indian 10-digit with optional +91 prefix
        r"(?:\+91[\s\-]?)?[6-9]\d{9}"
        r"|"
        # International: +CC then structured number (requires non-digit end)
        r"\+\d{1,3}[\s\-]\(?\d{2,4}\)?[\s\-]\d{3,4}[\s\-]\d{3,4}"
        r"|"
        # US/Canada with separators: (XXX) XXX-XXXX or XXX-XXX-XXXX or XXX.XXX.XXXX
        # Requires separators to be consistent (all same char) to avoid matching decimals
        r"\(?\d{3}\)[\s]\d{3}[\s\-]\d{4}"               # (XXX) XXX-XXXX
        r"|"
        r"\d{3}\-\d{3}\-\d{4}"                           # XXX-XXX-XXXX
        r")"
        r"(?!\d)",                                         # no digit immediately after
    ),

    "AADHAAR": re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)"),  # requires spaces as separators
    "PAN":     re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),

    # Personal URLs: exclude known academic/public domains
    "URL_PERSONAL": re.compile(r"https?://[^\s<>\"]+|www\.[a-zA-Z0-9\-]+\.[^\s<>\"]+"),

    # ID numbers: 8-12 digits only (6-7 digit numbers are too common as years, counts, etc.)
    # Must be isolated — no adjacent dots, slashes, letters, or hyphens
    "ID_NUM": re.compile(r"(?<![.\-/A-Za-z])\b\d{8,12}\b(?![.\-/A-Za-z])"),

    # Street address: house number must be followed by a named street with a clear street-type word.
    # Removed overly generic keywords: Main, Way, Cross, Block, Phase, Square, Extension, Circle.
    "STREET_ADDRESS": re.compile(
        r"(?<![A-Za-z\-/])"
        r"\b(?!REC-|REF-|ID-|DOC-|NO-|SR-|SL-|FIG-|EQ-|SEC-|CH-)"
        r"(?:\d{2,5}[A-Za-z]?|\d[A-Za-z])"               # house number: 2+ digits or 1 digit + letter
        r"\s+"
        r"(?:[A-Z][a-z]+\s+){1,3}"                        # 1-3 title-case street-name words (required, not optional)
        r"(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Lane|Ln\.|"
        r"Boulevard|Blvd\.|Drive|Dr\.|Court|Ct\.|Place|Pl\.|"
        r"Nagar|Marg|Chowk|Colony|Layout|Sector|"
        r"Highway|Hwy\.|Terrace|Ter\.)"                   # only unambiguous street-type words
        r"(?:,\s*(?:Apt|Suite|Unit|Floor|Flat|#)\s*\w+)?"
        r"(?:,\s*[A-Za-z][A-Za-z\s]{2,25})?",
        re.IGNORECASE,
    ),
}

# ── False positive blocklist for general NER ─────────────────────────────────
_PERSON_BLOCKLIST = {
    # Figure / table references
    "fig", "figure", "table", "eq", "equation", "algorithm",
    # Medical / scientific terms
    "std", "max", "min", "mean", "median", "sensitivity", "specificity",
    "glucose", "hemoglobin", "albumin", "creatinine", "insulin", "diabetes",
    "kidney", "renal", "blood", "urine", "protein", "bmi", "hba1c",
    "glycated", "hypertension", "cholesterol", "triglyceride",
    # ML / tech terms
    "catboost", "lightgbm", "xgboost", "adaboost", "python", "pandas",
    "matplotlib", "sklearn", "smote", "shap", "auc", "roc",
    "random", "forest", "gradient", "boosting", "machine", "learning",
    "artificial", "intelligence", "neural", "network", "model",
    # Common English words spaCy misclassifies as PERSON
    "lane", "ward", "reed", "rose", "king", "dean", "duke",
    "page", "nash", "bell", "fox", "gold", "hunter", "wolf",
    "field", "hill", "wood", "stone", "ford", "brook", "bridge",
    # Common false positives
    "gender", "male", "female", "age", "stage", "class", "type",
    "generative", "explainable", "rwanda", "rwandan",
    "challenges", "results", "discussion", "conclusion", "introduction",
    "related", "work", "system", "design", "data", "collection",
    "future", "scope", "review", "literature", "experimental",
    "overview", "background", "methodology", "implementation",
    "evaluation", "analysis", "comparison", "performance",
    # Single letters / abbreviations
    "al", "et", "vs", "tab", "sec", "ref",
    # Truncated words (PDF extraction artifacts)
    "specifici", "lymphocy", "clin",
    # Document section / field labels
    "aadhaar", "email", "username", "usernames", "phone", "address",
    "name", "date", "signature", "designation", "department",
    "subject", "reference", "summary", "section", "chapter",
    "appendix", "annexure", "enclosure",
    # Institutional / organisational words
    "faculty", "school", "college", "university", "institute",
    "department", "division", "centre", "center", "office",
    "committee", "board", "council", "bureau", "ministry",
    "finance", "science", "engineering", "medicine", "arts",
    "technology", "management", "commerce", "research",
    # Placeholder / generic table values
    "unknown", "n/a", "nil", "none", "na",
    "entity", "record", "handle", "contact", "identifier",
    # Indian cities / states / localities
    "haryana", "bangalore", "bengaluru", "mumbai", "delhi", "chennai",
    "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow",
    "bandra", "andheri", "powai", "juhu", "worli",
    "palm", "koramangala", "indiranagar", "whitefield",
    "gurgaon", "noida", "faridabad", "ghaziabad", "thane",
    "chandigarh", "mysore", "coimbatore", "vizag", "visakhapatnam",
    "greater", "south", "north", "east", "west", "new", "old",
    # Status / review labels
    "pending", "approved", "rejected", "completed", "closed", "open",
    # Document metadata labels
    "metadata", "version", "revision", "draft", "final", "confidential",
    "checklist", "matrix", "target", "entities",
    # Prepositions / conjunctions that appear in institutional names
    "of", "and", "for", "the", "in", "at", "by", "to",
}


def _is_valid_person(text: str) -> bool:
    """Return False if this PERSON entity is likely a false positive."""
    cleaned = text.strip().lower().rstrip('.,:;)')
    words = cleaned.split()

    # Full phrase in blocklist
    if cleaned in _PERSON_BLOCKLIST:
        return False

    # Any individual word in blocklist — strip all common punctuation
    if any(w.strip('.,;:()[]"\'-') in _PERSON_BLOCKLIST for w in words):
        return False

    # Single-word entities under 5 chars (too short and ambiguous)
    if len(words) == 1 and len(cleaned) < 5:
        return False

    # All uppercase (acronym)
    if text.strip().isupper():
        return False

    # Contains digits
    if any(c.isdigit() for c in text):
        return False

    # Looks like a section heading (more than 4 words is unlikely to be a name)
    if len(words) > 4:
        return False

    return True


# ── Lazy-load general NER model ───────────────────────────────────────────────
MODEL_DIR    = "./model/pii_ner_model"
_general_nlp = None


def _get_general_nlp():
    global _general_nlp
    if _general_nlp is None:
        try:
            _general_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except OSError:
            _general_nlp = False
    return _general_nlp if _general_nlp else None


def load_model():
    return spacy.load(MODEL_DIR)


def _is_excluded_url(url: str) -> bool:
    """Return True if the URL belongs to an academic/public domain and should not be redacted."""
    if _EXCLUDE_URL_DOMAINS.search(url):
        return True
    if "doi.org" in url.lower():
        return True
    # Too short to be a real personal URL
    bare = url.rstrip('/')
    if len(bare) < 14:
        return True
    return False


def _collect_regex_matches(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            matched = m.group()

            if label == "URL_PERSONAL" and _is_excluded_url(matched):
                continue

            # Skip ID_NUM that sit next to dots or slashes (DOI / version strings)
            if label == "ID_NUM":
                ctx = text[max(0, m.start() - 3):m.end() + 3]
                if re.search(r'[./]', ctx):
                    continue

            # Skip PHONE_NUM that are part of a longer pure-digit string
            if label == "PHONE_NUM":
                ctx_before = text[max(0, m.start() - 1):m.start()]
                ctx_after  = text[m.end():m.end() + 1]
                if (ctx_before and ctx_before.isdigit()) or (ctx_after and ctx_after.isdigit()):
                    continue

            spans.append((m.start(), m.end(), label))
    return spans


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """
    Merge overlapping spans. NER spans take priority over regex spans;
    within the same source, earlier start position wins.
    The input list is assumed ordered: NER spans first, regex spans last.
    """
    merged, occupied = [], []
    for s, e, l in sorted(spans, key=lambda x: x[0]):
        if not any(s < oe and e > os for os, oe in occupied):
            merged.append((s, e, l))
            occupied.append((s, e))
    return merged


def detect_pii(nlp, text: str) -> list[tuple[int, int, str]]:
    """
    Combine detections from:
    1. Custom PII NER model (entity_ruler + ner)
    2. en_core_web_sm PERSON entities
    3. Author-block heuristic: name on the line immediately above an email
    4. Regex fallback for structured entities
    """
    # ── Layer 1: Custom NER model ─────────────────────────────────────────────
    doc = nlp(text)
    ner_spans = []
    for ent in doc.ents:
        s, e, l = ent.start_char, ent.end_char, ent.label_
        snippet = text[s:e]
        if l == "URL_PERSONAL" and _is_excluded_url(snippet):
            continue
        # Reject suspiciously short custom-NER hits (likely tokenisation artifacts)
        if len(snippet.strip()) < 2:
            continue
        ner_spans.append((s, e, l))

    # ── Layer 2: General PERSON detection (en_core_web_sm) ───────────────────
    general_spans = []
    general_nlp = _get_general_nlp()
    if general_nlp:
        gdoc = general_nlp(text)
        for ent in gdoc.ents:
            if ent.label_ == "PERSON" and _is_valid_person(ent.text):
                general_spans.append((ent.start_char, ent.end_char, "NAME_STUDENT"))

    # ── Layer 3: Author-block heuristic ──────────────────────────────────────
    # A line that contains only an email, and whose immediately preceding
    # non-blank line is 2–4 alphabetic words, is treated as "name\nemail".
    email_re   = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}")
    name_above = []
    lines      = text.split("\n")

    # Build a char-offset map for each line so we can find the right position
    # (avoid text.find() which returns the first occurrence, not the right one)
    line_offsets = []
    pos = 0
    for ln in lines:
        line_offsets.append(pos)
        pos += len(ln) + 1  # +1 for the \n

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not email_re.fullmatch(line_stripped):
            continue
        # Walk back to find the nearest non-blank previous line
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j < 0:
            continue
        prev = lines[j].strip()
        words = prev.split()
        # Must be 2-4 purely alphabetic words (hyphens/apostrophes allowed)
        if not (2 <= len(words) <= 4):
            continue
        if not all(w.replace("-", "").replace("'", "").isalpha() for w in words):
            continue
        # Must pass blocklist check
        lower_words = [w.lower().strip('.,;:') for w in words]
        if any(w in _PERSON_BLOCKLIST for w in lower_words):
            continue
        if not _is_valid_person(prev):
            continue
        # Use the correct char offset for this specific line occurrence
        start = line_offsets[j] + lines[j].index(prev)
        end   = start + len(prev)
        name_above.append((start, end, "NAME_STUDENT"))

    # ── Layer 4: Regex ────────────────────────────────────────────────────────
    regex_spans = _collect_regex_matches(text)

    # NER spans go first so they win overlap resolution over regex
    all_spans = ner_spans + general_spans + name_above + regex_spans
    return _merge_spans(all_spans)


def _extract_page_text(page) -> str:
    """Extract text from page; fall back to OCR for image-only pages."""
    text = page.get_text("text").strip()
    if text:
        return page.get_text("text")
    if OCR_AVAILABLE:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img)
    return ""


def redact_pdf(pdf_bytes: bytes, nlp) -> tuple[bytes, dict]:
    """
    Redact PII from a PDF. Returns (redacted_bytes, entity_count_dict).
    """
    src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    entity_counts: dict[str, int] = {}

    for page in src_doc:
        page_text = _extract_page_text(page)
        if not page_text.strip():
            continue

        spans = detect_pii(nlp, page_text)

        for start, end, label in spans:
            snippet = page_text[start:end].strip()
            if not snippet or len(snippet) < 2:
                continue

            # Try exact phrase search first
            quads = page.search_for(snippet, quads=True)

            # Fallback: for multi-word entities, try each word — but only if the
            # word is long enough to be unique (≥6 chars) to avoid page-wide
            # collateral redaction of common short words.
            if not quads:
                for word in snippet.split():
                    word_clean = word.strip('.,;:()')
                    if len(word_clean) >= 6:
                        word_quads = page.search_for(word_clean, quads=True)
                        quads.extend(word_quads)

            for quad in quads:
                page.add_redact_annot(quad, fill=(0, 0, 0))

            if quads:
                entity_counts[label] = entity_counts.get(label, 0) + 1

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    src_doc.set_metadata({})
    buf = io.BytesIO()
    src_doc.save(buf, garbage=4, deflate=True, clean=True)
    src_doc.close()

    return buf.getvalue(), entity_counts
