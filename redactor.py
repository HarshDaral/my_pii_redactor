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

# ── Regex patterns ────────────────────────────────────────────────────────────
# DOI and known academic/public domains to exclude from URL redaction
_EXCLUDE_URL_DOMAINS = re.compile(
    r"https?://(doi\.org|dx\.doi\.org|arxiv\.org|pubmed\.ncbi|scholar\.google|"
    r"www\.ncbi\.nlm|orcid\.org|creativecommons\.org|ieee\.org|springer\.com|"
    r"elsevier\.com|wiley\.com|acm\.org|researchgate\.net|academia\.edu|"
    r"github\.com|linkedin\.com/company|twitter\.com|youtube\.com|wikipedia\.org|"
    # Government domains
    r"[a-z0-9\-]+\.gov\.in|[a-z0-9\-]+\.gov|[a-z0-9\-]+\.nic\.in|"
    r"[a-z0-9\-]+\.mil|[a-z0-9\-]+\.edu|[a-z0-9\-]+\.ac\.in|"
    r"[a-z0-9\-]+\.res\.in|[a-z0-9\-]+\.org\.in)",
    re.IGNORECASE
)

REGEX_PATTERNS = {
    "EMAIL":        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),

    # ── Phone numbers ────────────────────────────────────────────────────────
    # Covers: Indian mobile, international +CC format, US/UK with dashes/dots/spaces
    "PHONE_NUM":    re.compile(
        r"(?:"
        # Indian 10-digit (with or without +91 / 0 prefix)
        r"(?:\+91[\s\-]?)?[6-9]\d{9}"
        r"|"
        # International: +<country_code> <number>, e.g. +1 415 555 2671
        r"\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}"
        r"|"
        # US/Canada: (XXX) XXX-XXXX or XXX-XXX-XXXX or XXX.XXX.XXXX
        r"\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}"
        r"|"
        # Compact 10-digit (no separators) starting with common prefixes
        r"(?:0\d{10}|\d{5}[\s\-]\d{5})"
        r")",
        re.VERBOSE
    ),

    "AADHAAR":      re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN":          re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "URL_PERSONAL": re.compile(r"https?://[^\s]+|www\.[^\s]+"),
    "ID_NUM":       re.compile(r"\b\d{6,12}\b"),

    # ── Street address patterns ──────────────────────────────────────────────
    # Strict requirements to avoid section headings:
    # - House number must be 2+ digits OR have a letter suffix (42, 14B, 100)
    # - Street type keyword must follow within 5 words
    # - Single digit section numbers (1, 2, 3...) are excluded
    "STREET_ADDRESS": re.compile(
        r"(?<![A-Za-z\-/])"
        r"\b(?!REC-|REF-|ID-|DOC-|NO-|SR-|SL-)"
        r"(?:\d{2,5}[A-Za-z]?|\d[A-Za-z])"            # 2+ digit house number OR 1 digit + letter (e.g. 5B)
        r"\s+"
        r"(?:[A-Z][a-z]+\s+){0,3}"                     # 0-3 title-case words (street name)
        r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Lane|Ln\.?|"
        r"Boulevard|Blvd\.?|Drive|Dr\.?|Court|Ct\.?|Place|Pl\.?|"
        r"Nagar|Marg|Chowk|Colony|Layout|Cross|Main|"
        r"Sector|Block|Phase|Extension|Highway|Hwy\.?|"
        r"Circle|Cir\.?|Square|Sq\.?|Way|Terrace|Ter\.?)"
        r"(?:,\s*(?:Apt|Suite|Unit|Floor|Flat|#)\s*\w+)?"
        r"(?:,\s*[A-Za-z][A-Za-z\s]{2,25})?",
        re.IGNORECASE
    ),
}

# ── False positive blocklist for general NER ─────────────────────────────────
# Terms en_core_web_sm commonly misclassifies as PERSON in academic/medical docs
_PERSON_BLOCKLIST = {
    # Figure/table references
    "fig", "figure", "table", "eq", "equation", "algorithm",
    # Medical/scientific terms
    "std", "max", "min", "mean", "median", "sensitivity", "specificity",
    "glucose", "hemoglobin", "albumin", "creatinine", "insulin", "diabetes",
    "kidney", "renal", "blood", "urine", "protein", "bmi", "hba1c",
    "glycated", "hypertension", "cholesterol", "triglyceride",
    # ML/tech terms
    "catboost", "lightgbm", "xgboost", "adaboost", "python", "pandas",
    "matplotlib", "sklearn", "smote", "shap", "auc", "roc",
    "random", "forest", "gradient", "boosting", "machine", "learning",
    "artificial", "intelligence", "neural", "network", "model",
    # Common false positives
    "gender", "male", "female", "age", "stage", "class", "type",
    "generative", "explainable", "rwanda", "rwandan",
    "challenges", "results", "discussion", "conclusion", "introduction",
    # Single letters / abbreviations
    "al", "et", "vs", "fig", "tab", "sec", "ref",
    # Truncated words (PDF extraction artifacts)
    "specifici", "lymphocy", "clin",
    # Document section/field labels that get mistagged as names
    "aadhaar", "email", "username", "usernames", "phone", "address",
    "name", "date", "signature", "designation", "department",
    "subject", "reference", "summary", "conclusion", "introduction",
    "section", "chapter", "appendix", "annexure", "enclosure",
    # Placeholder/generic table values
    "unknown", "unknown user", "n/a", "nil", "none", "na",
    "entity", "record", "handle", "contact", "identifier",
    # Indian cities/states/localities commonly misidentified as names
    "haryana", "bangalore", "bengaluru", "mumbai", "delhi", "chennai",
    "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow",
    "bandra", "bandra west", "andheri", "powai", "juhu", "worli",
    "palm meadows", "koramangala", "indiranagar", "whitefield",
    "gurgaon", "noida", "faridabad", "ghaziabad", "thane", "navi mumbai",
    "chandigarh", "mysore", "coimbatore", "vizag", "visakhapatnam",
    # Status/review labels
    "pending", "pending review", "approved", "rejected", "under review",
    "completed", "in progress", "closed", "open",
    # Document metadata labels
    "metadata", "version", "revision", "draft", "final", "confidential",
    "target entities", "checklist", "overview", "matrix",
}

def _is_valid_person(text: str) -> bool:
    """Return False if this PERSON entity is likely a false positive."""
    cleaned = text.strip().lower().rstrip('.')
    words = cleaned.split()

    # Full phrase in blocklist (e.g. "palm meadows", "pending review")
    if cleaned in _PERSON_BLOCKLIST:
        return False

    # Any individual word in blocklist
    if any(w.strip('.,') in _PERSON_BLOCKLIST for w in words):
        return False

    # Single-word entities under 4 chars
    if len(words) == 1 and len(cleaned) < 4:
        return False

    # All uppercase (likely an acronym)
    if text.strip().isupper():
        return False

    # Contains digits
    if any(c.isdigit() for c in text):
        return False

    return True

# General-purpose model for PERSON detection across all document types
MODEL_DIR   = "./model/pii_ner_model"
_general_nlp = None

def _get_general_nlp():
    """Lazy-load en_core_web_sm for broad PERSON detection."""
    global _general_nlp
    if _general_nlp is None:
        try:
            _general_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except OSError:
            _general_nlp = False  # not installed, skip
    return _general_nlp if _general_nlp else None


def load_model():
    return spacy.load(MODEL_DIR)


def _collect_regex_matches(text: str) -> list[tuple[int, int, str]]:
    spans = []
    for label, pattern in REGEX_PATTERNS.items():
        for m in pattern.finditer(text):
            matched = m.group()
            # Skip DOI and known academic/public URLs (use search not match)
            if label == "URL_PERSONAL" and _EXCLUDE_URL_DOMAINS.search(matched):
                continue
            # Also skip bare doi.org fragments and very short URLs
            if label == "URL_PERSONAL" and (
                "doi.org" in matched.lower() or
                len(matched.rstrip('/')) < 12   # too short to be a real personal URL
            ):
                continue
            # Skip DOI-style ID numbers (contain dots or slashes like 10.1145/...)
            if label == "ID_NUM" and re.search(r'[./]', text[max(0,m.start()-5):m.end()+5]):
                continue
            spans.append((m.start(), m.end(), label))
    return spans


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    merged, occupied = [], []
    for s, e, l in sorted(spans, key=lambda x: x[0]):
        if not any(s < oe and e > os for os, oe in occupied):
            merged.append((s, e, l))
            occupied.append((s, e))
    return merged


def detect_pii(nlp, text: str) -> list[tuple[int, int, str]]:
    """
    Combine detections from:
    1. Custom PII NER model
    2. en_core_web_sm PERSON entities (catches names in academic/formal docs)
    3. Names appearing directly above email addresses (author block pattern)
    4. Regex fallback for structured entities
    """
    # Custom model
    doc = nlp(text)
    ner_spans = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

    # Filter out EntityRuler URL false positives (DOIs etc.) post-detection
    cleaned_ner = []
    for s, e, l in ner_spans:
        snippet = text[s:e]
        if l == "URL_PERSONAL" and (
            "doi.org" in snippet.lower() or
            _EXCLUDE_URL_DOMAINS.search(snippet) or
            len(snippet.rstrip('/')) < 12
        ):
            continue
        cleaned_ner.append((s, e, l))
    ner_spans = cleaned_ner
    general_nlp = _get_general_nlp()
    general_spans = []
    if general_nlp:
        gdoc = general_nlp(text)
        for ent in gdoc.ents:
            if ent.label_ == "PERSON" and _is_valid_person(ent.text):
                general_spans.append((ent.start_char, ent.end_char, "NAME_STUDENT"))

    # Author-block pattern: line directly above an email is likely a name
    # Matches: "Firstname Lastname\nemail@domain.com"
    email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    name_above_email = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if email_re.fullmatch(line_stripped) and i > 0:
            prev = lines[i - 1].strip()
            words = prev.split()
            if 1 < len(words) <= 5 and all(w.replace("-","").replace("'","").isalpha() for w in words):
                idx = text.find(prev)
                if idx != -1:
                    name_above_email.append((idx, idx + len(prev), "NAME_STUDENT"))

    # Table name pattern: lines that look like full names (2-4 capitalized words)
    # appearing in table rows — catches names in structured tables
    table_names = []
    name_line_re = re.compile(
        r"^(?:Dr\.?\s+|Mr\.?\s+|Mrs\.?\s+|Ms\.?\s+|Prof\.?\s+|Er\.?\s+)?"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}$"   # 2–5 capitalized words
    )
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if name_line_re.match(line_stripped):
            # Make sure it's not already caught and not a known false positive
            if not any(_is_valid_person.__wrapped__ if hasattr(_is_valid_person, '__wrapped__') else
                       line_stripped.lower() in _PERSON_BLOCKLIST for _ in [1]):
                lower_words = [w.lower().strip('.,') for w in line_stripped.split()]
                if not any(w in _PERSON_BLOCKLIST for w in lower_words):
                    idx = text.find(line_stripped)
                    if idx != -1:
                        table_names.append((idx, idx + len(line_stripped), "NAME_STUDENT"))

    # Regex
    regex_spans = _collect_regex_matches(text)

    all_spans = ner_spans + general_spans + name_above_email + table_names + regex_spans
    return _merge_spans(all_spans)


def _extract_page_text(page) -> str:
    """Extract text; fall back to OCR for scanned pages."""
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
    Redact PII from a PDF.
    Returns redacted bytes + entity count dict.
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

            # Try exact phrase match
            quads = page.search_for(snippet, quads=True)

            # Fallback: search each word individually for multi-word entities
            # (handles line-break splits and ligature issues)
            if not quads:
                for word in snippet.split():
                    if len(word) > 2:  # skip very short words
                        word_quads = page.search_for(word, quads=True)
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
