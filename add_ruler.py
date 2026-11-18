"""
add_ruler.py  -  Adds/updates EntityRuler in the saved model.
Excludes DOI and academic URLs from URL_PERSONAL detection.
Run: python add_ruler.py
"""
import spacy

MODEL_DIR = "./model/pii_ner_model"

print(f"Loading model from {MODEL_DIR} …")
nlp = spacy.load(MODEL_DIR)

# Remove existing ruler to rebuild cleanly
if "entity_ruler" in nlp.pipe_names:
    nlp.remove_pipe("entity_ruler")

ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})

# Personal URLs — exclude academic/public domains via negative lookahead.
# Pattern covers both https:// and www. prefixes consistently.
PERSONAL_URL_PATTERN = (
    r"https?://(?!(?:doi\.org|dx\.doi\.org|arxiv\.org|pubmed|ncbi|orcid\.org|"
    r"creativecommons|ieee\.org|springer|elsevier|wiley|acm\.org|"
    r"researchgate|academia\.edu|github\.com|wikipedia|"
    r"[a-z0-9\-]+\.gov|[a-z0-9\-]+\.edu|[a-z0-9\-]+\.ac\.in))[^\s]+"
)

patterns = [
    # Email — require proper TLD (2-6 alpha chars only)
    {"label": "EMAIL", "pattern": [{"TEXT": {"REGEX": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}"}}]},

    # Phone — Indian 10-digit mobile (starts 6-9), no prefix
    {"label": "PHONE_NUM", "pattern": [{"TEXT": {"REGEX": r"^[6-9]\d{9}$"}}]},
    # Indian with +91 prefix (two or three tokens: '+91' '9876543210' or '+' '91' '9876543210')
    {"label": "PHONE_NUM", "pattern": [{"TEXT": "+91"}, {"TEXT": {"REGEX": r"^[6-9]\d{9}$"}}]},
    {"label": "PHONE_NUM", "pattern": [{"TEXT": "+"}, {"TEXT": "91"}, {"TEXT": {"REGEX": r"^[6-9]\d{9}$"}}]},
    # +91 with 5+5 split: '+91' '98765' '43210'
    {"label": "PHONE_NUM", "pattern": [{"TEXT": "+91"}, {"TEXT": {"REGEX": r"^[6-9]\d{4}$"}}, {"TEXT": {"REGEX": r"^\d{5}$"}}]},
    # US format (XXX) XXX-XXXX
    {"label": "PHONE_NUM", "pattern": [{"TEXT": {"REGEX": r"^\(\d{3}\)$"}}, {"TEXT": {"REGEX": r"^\d{3}-\d{4}$"}}]},
    # Dash-separated XXX-XXX-XXXX
    {"label": "PHONE_NUM", "pattern": [{"TEXT": {"REGEX": r"^\d{3}-\d{3}-\d{4}$"}}]},

    # Personal URLs
    {"label": "URL_PERSONAL", "pattern": [{"TEXT": {"REGEX": PERSONAL_URL_PATTERN}}]},

    # Aadhaar — requires spaces between groups (4 4 4)
    {"label": "ID_NUM", "pattern": [
        {"TEXT": {"REGEX": r"^\d{4}$"}},
        {"TEXT": {"REGEX": r"^\d{4}$"}},
        {"TEXT": {"REGEX": r"^\d{4}$"}},
    ]},
    # Aadhaar — 12 digits compact (no spaces)
    {"label": "ID_NUM", "pattern": [{"TEXT": {"REGEX": r"^\d{12}$"}}]},
    # PAN card
    {"label": "ID_NUM", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{5}[0-9]{4}[A-Z]$"}}]},
    # 8-10 digit IDs (raised floor from 6 to 8 to avoid matching common numbers)
    {"label": "ID_NUM", "pattern": [{"TEXT": {"REGEX": r"^\d{8,10}$"}}]},
]

ruler.add_patterns(patterns)
nlp.to_disk(MODEL_DIR)
print(f"Updated EntityRuler saved → {MODEL_DIR}")
print(f"Pipes: {nlp.pipe_names}")
