"""
train_incremental.py  -  Continue training the existing model on docs 500-1000.
Loads ./model/pii_ner_model and fine-tunes it, then saves back to the same path.
"""
import json, random
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import compounding

DATA_PATH  = "./data/train.json"
MODEL_DIR  = "./model/pii_ner_model"
DOC_START  = 2500
DOC_END    = 3000
ITERATIONS = 15
DROPOUT    = 0.2
SEED       = 42

PII_LABELS = [
    "NAME_STUDENT", "EMAIL", "USERNAME",
    "ID_NUM", "PHONE_NUM", "URL_PERSONAL", "STREET_ADDRESS",
]

random.seed(SEED)

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
            s = offsets[i]; e = offsets[i] + len(tokens[i]); i += 1
            while i < len(labels) and labels[i] == f"I-{ent}":
                e = offsets[i] + len(tokens[i]); i += 1
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

def main():
    print(f"Loading existing model from {MODEL_DIR} …")
    nlp = spacy.load(MODEL_DIR)
    print(f"Existing labels: {nlp.get_pipe('ner').labels}")

    print(f"\nLoading docs [{DOC_START}:{DOC_END}] from {DATA_PATH} …")
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    new_docs = raw[DOC_START:DOC_END]
    pii_count = sum(1 for d in new_docs if any(l != "O" for l in d["labels"]))
    print(f"New docs: {len(new_docs)}  |  Docs with PII: {pii_count}")

    # Build examples
    examples = []
    for doc in new_docs:
        text, spans = bio_to_spans(doc["tokens"], doc["labels"], doc["trailing_whitespace"])
        spacy_doc = nlp.make_doc(text)
        try:
            examples.append(Example.from_dict(spacy_doc, {"entities": dedup(spans)}))
        except Exception:
            pass

    print(f"Valid examples: {len(examples)}")

    # Use docs 2000-2500 as dev set (already trained on those)
    dev_raw = [d for d in raw[2000:2500] if any(l != "O" for l in d["labels"])][:50]
    dev_ex  = []
    for doc in dev_raw:
        text, spans = bio_to_spans(doc["tokens"], doc["labels"], doc["trailing_whitespace"])
        spacy_doc = nlp.make_doc(text)
        try:
            dev_ex.append(Example.from_dict(spacy_doc, {"entities": dedup(spans)}))
        except Exception:
            pass

    print(f"Dev examples (for eval): {len(dev_ex)}\n")

    # ── Fine-tune ─────────────────────────────────────────────────────────
    best_f1, best_bytes = 0.0, None
    other = [p for p in nlp.pipe_names if p != "ner"]
    batch_sizes = compounding(4.0, 32.0, 1.001)

    with nlp.disable_pipes(*other):
        for itn in range(ITERATIONS):
            random.shuffle(examples)
            losses = {}
            for batch in spacy.util.minibatch(examples, size=batch_sizes):
                nlp.update(batch, drop=DROPOUT, losses=losses)

            sc   = nlp.evaluate(dev_ex)
            f1   = sc.get("ents_f", 0.0)
            p    = sc.get("ents_p", 0.0)
            r    = sc.get("ents_r", 0.0)
            mark = ""
            if f1 >= best_f1:
                best_f1    = f1
                best_bytes = nlp.to_bytes()
                mark       = "  ← best"
            print(f"  Iter {itn+1:>2}/{ITERATIONS}  loss={losses.get('ner',0):>9.1f}"
                  f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}{mark}")

    # Restore best and save
    if best_bytes:
        nlp.from_bytes(best_bytes)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"\nModel saved → {MODEL_DIR}  (best F1={best_f1:.3f})")

    # Final per-label breakdown
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
