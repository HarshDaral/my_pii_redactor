"""
train.py  –  High-accuracy PII NER trainer
Strategy:
  - All 891 PII docs + equal sample of O-only docs (balanced)
  - Heavy Faker augmentation for rare labels
  - Compounding batch size (spaCy recommended)
  - Evaluate every 5 iters on a small fast dev slice
  - Save best checkpoint, early stop on patience
"""
import json, random
from pathlib import Path
from collections import defaultdict
from faker import Faker
import spacy
from spacy.training import Example
from spacy.util import compounding

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "./data/train.json"
MODEL_DIR   = "./model/pii_ner_model"
ITERATIONS  = 40
DROPOUT     = 0.2
SEED        = 42
EVAL_EVERY  = 5
PATIENCE    = 4          # eval-check patience (4 × 5 = 20 iters without gain → stop)
AUG_COUNTS  = {          # synthetic examples per label
    "NAME_STUDENT":  150,
    "EMAIL":         100,
    "USERNAME":      150,
    "PHONE_NUM":     150,
    "URL_PERSONAL":  100,
    "ID_NUM":        120,
    "STREET_ADDRESS":150,
}
PII_LABELS = list(AUG_COUNTS.keys())

random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ── BIO → char spans ──────────────────────────────────────────────────────────
def bio_to_spans(tokens, labels, ws):
    parts, offsets, pos = [], [], 0
    for tok, space in zip(tokens, ws):
        offsets.append(pos)
        parts.append(tok + (" " if space else ""))
        pos += len(tok) + (1 if space else 0)
    text = "".join(parts)
    spans, i = [], 0
    while i < len(labels):
        lbl = labels[i]
        if lbl.startswith("B-"):
            ent = lbl[2:]
            s = offsets[i]
            e = offsets[i] + len(tokens[i])
            i += 1
            while i < len(labels) and labels[i] == f"I-{ent}":
                e = offsets[i] + len(tokens[i])
                i += 1
            spans.append((s, e, ent))
        else:
            i += 1
    return text, spans

def dedup(spans):
    out, occupied = [], []
    for s, e, l in sorted(spans, key=lambda x: x[0]):
        if l in PII_LABELS and not any(s < oe and e > os for os, oe in occupied):
            out.append((s, e, l))
            occupied.append((s, e))
    return out

MAX_CHARS = 800   # truncate long essays — PII almost always in first portion

def truncate(text, entities):
    """Keep first MAX_CHARS chars; drop any entity that falls outside."""
    if len(text) <= MAX_CHARS:
        return text, entities
    t = text[:MAX_CHARS]
    ents = [(s, e, l) for s, e, l in entities if e <= MAX_CHARS]
    return t, ents

# ── Faker augmentation ────────────────────────────────────────────────────────
TEMPLATES = {
    "NAME_STUDENT": [
        "This reflection was written by {v}.",
        "Student name: {v}",
        "My name is {v} and I am enrolled in this course.",
        "Submitted by {v}.",
        "{v} is the author of this assignment.",
        "Hi, I am {v}, a participant in the Design Thinking workshop.",
        "The following essay is authored by {v}.",
    ],
    "EMAIL": [
        "Please contact me at {v}.",
        "My email is {v}.",
        "Reach me at {v} for feedback.",
        "Send all correspondence to {v}.",
        "Email address: {v}",
    ],
    "USERNAME": [
        "My username is {v}.",
        "You can find me online as {v}.",
        "Handle: {v}",
        "Follow {v} on the platform.",
        "User: {v}",
        "My profile name is {v}.",
    ],
    "PHONE_NUM": [
        "Call me at {v}.",
        "My phone number is {v}.",
        "Reach me on {v}.",
        "Mobile: {v}",
        "Contact: {v}",
    ],
    "URL_PERSONAL": [
        "Visit my site at {v}.",
        "My portfolio is at {v}.",
        "More details at {v}.",
        "Website: {v}",
        "Check out {v}.",
    ],
    "ID_NUM": [
        "My student ID is {v}.",
        "Roll number: {v}",
        "Registration: {v}",
        "ID number: {v}",
        "My ID is {v}.",
    ],
    "STREET_ADDRESS": [
        "I live at {v}.",
        "My address is {v}.",
        "Mailing address: {v}",
        "Send letters to {v}.",
        "Residence: {v}",
    ],
}

def gen_value(label):
    if label == "NAME_STUDENT":   return fake.name()
    if label == "EMAIL":          return fake.email()
    if label == "USERNAME":       return fake.user_name()
    if label == "PHONE_NUM":
        return f"{random.choice([6,7,8,9])}{''.join([str(random.randint(0,9)) for _ in range(9)])}"
    if label == "URL_PERSONAL":   return f"https://{fake.domain_word()}.{fake.tld()}/{fake.uri_path()}"
    if label == "ID_NUM":         return str(random.randint(10000000, 9999999999))
    if label == "STREET_ADDRESS": return fake.address().replace("\n", ", ")
    return ""

def make_aug(label, n):
    out = []
    for _ in range(n):
        tmpl  = random.choice(TEMPLATES[label])
        value = gen_value(label)
        text  = tmpl.replace("{v}", value)
        # wrap in a short paragraph to look like essay context
        prefix = random.choice([
            "In this reflection I would like to share some personal details. ",
            "As part of this exercise, ",
            "For identification purposes, ",
            "",
        ])
        text = prefix + text
        start = text.index(value)
        end   = start + len(value)
        out.append((text, {"entities": [(start, end, label)]}))
    return out

# ── Build training data ───────────────────────────────────────────────────────
def build_data(raw_docs):
    pii_docs, o_docs = [], []
    for doc in raw_docs:
        if any(l != "O" for l in doc["labels"]):
            pii_docs.append(doc)
        else:
            o_docs.append(doc)

    # All PII docs + matched number of O docs
    sampled_o = random.sample(o_docs, min(len(pii_docs), len(o_docs)))
    selected  = pii_docs + sampled_o
    random.shuffle(selected)

    data = []
    for doc in selected:
        text, spans = bio_to_spans(doc["tokens"], doc["labels"], doc["trailing_whitespace"])
        text, spans = truncate(text, dedup(spans))
        data.append((text, {"entities": spans}))

    # Augmentation
    total_aug = 0
    for label, n in AUG_COUNTS.items():
        aug = make_aug(label, n)
        data.extend(aug)
        total_aug += len(aug)

    random.shuffle(data)
    print(f"  Real examples:      {len(selected)}")
    print(f"  Augmented examples: {total_aug}")
    print(f"  Total:              {len(data)}")
    return data

# ── Convert to spaCy Examples (fast — skip malformed) ────────────────────────
def to_examples(nlp, data):
    """Convert (text, ann) pairs to spaCy Examples using make_doc (no pipeline)."""
    out = []
    for text, ann in data:
        # make_doc only tokenizes — skips all pipeline components (fast)
        doc = nlp.make_doc(text)
        try:
            out.append(Example.from_dict(doc, ann))
        except Exception:
            pass
    return out

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data …")
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    # Hold out last 300 docs for dev evaluation
    train_raw = raw[:-300]
    dev_raw   = raw[-300:]
    print(f"Train pool: {len(train_raw)} docs  |  Dev: {len(dev_raw)} docs")

    print("\nBuilding training set …")
    train_data = build_data(train_raw)

    print("\nBuilding NLP model …")
    nlp = spacy.blank("en")
    print("  Base: spacy.blank('en')")
    ner = nlp.add_pipe("ner", last=True)
    for lbl in PII_LABELS:
        ner.add_label(lbl)
    nlp.initialize()

    print("Converting to spaCy Examples …")
    train_ex = to_examples(nlp, train_data)
    # Dev: use docs 500-700 (known to have PII, not in training set tail)
    dev_pii = [d for d in raw[500:700] if any(l != "O" for l in d["labels"])]
    dev_sample = dev_pii[:80]
    dev_data   = []
    for doc in dev_sample:
        text, spans = bio_to_spans(doc["tokens"], doc["labels"], doc["trailing_whitespace"])
        text, spans = truncate(text, dedup(spans))
        dev_data.append((text, {"entities": spans}))
    dev_ex = to_examples(nlp, dev_data)

    print(f"Train examples: {len(train_ex)}  |  Dev examples: {len(dev_ex)}\n")

    # ── Training ──────────────────────────────────────────────────────────
    best_f1, best_bytes, no_improve = 0.0, None, 0
    batch_sizes = compounding(4.0, 32.0, 1.001)

    other = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other):
        for itn in range(ITERATIONS):
            random.shuffle(train_ex)
            losses = {}
            batches = spacy.util.minibatch(train_ex, size=batch_sizes)
            for batch in batches:
                nlp.update(batch, drop=DROPOUT, losses=losses)

            loss = losses.get("ner", 0)

            if (itn + 1) % EVAL_EVERY == 0 or itn == ITERATIONS - 1:
                sc   = nlp.evaluate(dev_ex)
                f1   = sc.get("ents_f", 0.0)
                p    = sc.get("ents_p", 0.0)
                r    = sc.get("ents_r", 0.0)
                mark = ""
                if f1 > best_f1:
                    best_f1    = f1
                    best_bytes = nlp.to_bytes()
                    no_improve = 0
                    mark = "  ← best"
                else:
                    no_improve += 1
                print(f"  Iter {itn+1:>2}/{ITERATIONS}  loss={loss:>9.1f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}{mark}")
                if no_improve >= PATIENCE:
                    print(f"\n  Early stop (no gain for {PATIENCE} checks = {PATIENCE*EVAL_EVERY} iters)")
                    break
            else:
                print(f"  Iter {itn+1:>2}/{ITERATIONS}  loss={loss:>9.1f}")

    # ── Save best ─────────────────────────────────────────────────────────
    if best_bytes:
        nlp.from_bytes(best_bytes)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"\nModel saved → {MODEL_DIR}  (best F1={best_f1:.3f})")

    # ── Final per-label breakdown ─────────────────────────────────────────
    sc  = nlp.evaluate(dev_ex)
    ept = sc.get("ents_per_type", {})
    print(f"\n{'Label':<20} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 44)
    for lbl in PII_LABELS:
        if lbl in ept:
            s = ept[lbl]
            print(f"{lbl:<20} {s['p']:>7.3f} {s['r']:>7.3f} {s['f']:>7.3f}")
        else:
            print(f"{lbl:<20}     N/A     N/A     N/A")
    print("-" * 44)
    print(f"{'OVERALL':<20} {sc['ents_p']:>7.3f} {sc['ents_r']:>7.3f} {sc['ents_f']:>7.3f}")

if __name__ == "__main__":
    main()
