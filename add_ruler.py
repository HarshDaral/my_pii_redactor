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

# Personal URLs only — explicitly excludes doi.org and academic domains
PERSONAL_URL_PATTERN = (
    r"https?://(?!(?:doi\.org|dx\.doi\.org|arxiv\.org|pubmed|ncbi|orcid\.org|"
    r"creativecommons|ieee\.org|springer|elsevier|wiley|acm\.org|"
    r"researchgate|academia\.edu|github\.com|wikipedia))[^\s]+"
)

patterns = [
    {"label": "EMAIL",        "pattern": [{"TEXT": {"REGEX": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+"}}]},
    {"label": "PHONE_NUM",    "pattern": [{"TEXT": {"REGEX": r"[6-9]\d{9}"}}]},
    {"label": "URL_PERSONAL", "pattern": [{"TEXT": {"REGEX": PERSONAL_URL_PATTERN}}]},
    {"label": "URL_PERSONAL", "pattern": [{"TEXT": {"REGEX": r"www\.[^\s]+"}}]},
    # Aadhaar 4-4-4 format
    {"label": "ID_NUM", "pattern": [{"TEXT": {"REGEX": r"\d{4}"}}, {"TEXT": {"REGEX": r"\d{4}"}}, {"TEXT": {"REGEX": r"\d{4}"}}]},
    {"label": "ID_NUM", "pattern": [{"TEXT": {"REGEX": r"\d{12}"}}]},
    # PAN card
    {"label": "ID_NUM", "pattern": [{"TEXT": {"REGEX": r"[A-Z]{5}[0-9]{4}[A-Z]"}}]},
    # Pure digit IDs (6-10 digits, not DOI-style)
    {"label": "ID_NUM", "pattern": [{"TEXT": {"REGEX": r"^\d{6,10}$"}}]},

    # Phone patterns for EntityRuler
    # Indian with +91 prefix
    {"label": "PHONE_NUM", "pattern": [{"TEXT": "+91"}, {"TEXT": {"REGEX": r"\d{10}"}}]},
    {"label": "PHONE_NUM", "pattern": [{"TEXT": {"REGEX": r"\+91"}, "OP": "?"}, {"TEXT": {"REGEX": r"[6-9]\d{9}"}}]},
    # US format (123) 456-7890
    {"label": "PHONE_NUM", "pattern": [{"TEXT": {"REGEX": r"\(\d{3}\)"}}, {"TEXT": {"REGEX": r"\d{3}-\d{4}"}}]},
    # Dash-separated 10-digit
    {"label": "PHONE_NUM", "pattern": [{"TEXT": {"REGEX": r"\d{3}-\d{3}-\d{4}"}}]},
    {"label": "PHONE_NUM", "pattern": [{"TEXT": {"REGEX": r"\d{5}-\d{5}"}}]},
]

ruler.add_patterns(patterns)
nlp.to_disk(MODEL_DIR)
print(f"Updated EntityRuler saved → {MODEL_DIR}")
print(f"Pipes: {nlp.pipe_names}")
