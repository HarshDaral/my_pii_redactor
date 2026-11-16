"""
evaluate.py - Evaluate the trained PII NER model.

Usage:
  python evaluate.py                          # uses held-out slice of train.json
  python evaluate.py --test path/to/test.json # uses your own test file (same format)
"""
import json
import sys
import spacy
from spacy.training import Example

MODEL_DIR  = "./model/pii_ner_model"
TRAIN_PATH = "./data/train.json"
DEV_START  = -300  # last 300 docs held out during training

PII_LABELS = [
    "NAME_STUDENT", "EMAIL", "USERNAME",
    "ID_NUM", "PHONE_NUM", "URL_PERSONAL", "STREET_ADDRESS",
]

def bio_to_spans(tokens, labels, ws):
    text_parts, offsets, char_pos = [], [], 0
    for tok, space in zip(tokens, ws):
        offsets.append(char_pos)
        text_parts.append(tok + (" " if space else ""))
        char_pos += len(tok) + (1 if space else 0)
    full_text = "".join(text_parts)
    spans, i = [], 0
    while i < len(labels):
        lbl = labels[i]
        if lbl.startswith("B-"):
            entity = lbl[2:]
            start = offsets[i]
            end   = offsets[i] + len(tokens[i])
            i += 1
            while i < len(labels) and labels[i] == f"I-{entity}":
                end = offsets[i] + len(tokens[i])
                i += 1
            spans.append((start, end, entity))
        else:
            i += 1
    return full_text, spans

def load_examples(nlp, raw_docs):
    examples = []
    for doc in raw_docs:
        text, spans = bio_to_spans(doc["tokens"], doc["labels"], doc["trailing_whitespace"])
        filtered = [(s, e, l) for s, e, l in spans if l in PII_LABELS]
        clean, occupied = [], []
        for s, e, l in filtered:
            if not any(s < oe and e > os for os, oe in occupied):
                clean.append((s, e, l))
                occupied.append((s, e))
        spacy_doc = nlp.make_doc(text)
        try:
            examples.append(Example.from_dict(spacy_doc, {"entities": clean}))
        except Exception:
            pass
    return examples

def main():
    # Parse optional --test flag
    test_path = None
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        test_path = sys.argv[idx + 1]

    print(f"Loading model from {MODEL_DIR} …")
    nlp = spacy.load(MODEL_DIR)

    if test_path:
        print(f"Loading test data from {test_path} …")
        with open(test_path, encoding="utf-8") as f:
            raw = json.load(f)
        source_label = f"external file ({len(raw)} docs)"
    else:
        print(f"Loading held-out slice [{DEV_START}:] from {TRAIN_PATH} …")
        with open(TRAIN_PATH, encoding="utf-8") as f:
            full = json.load(f)
        raw = full[DEV_START:]
        source_label = f"held-out slice (last 300 docs)"

    examples = load_examples(nlp, raw)
    print(f"Evaluating on {len(examples)} examples from {source_label} …\n")

    scores = nlp.evaluate(examples)
    ents_per_type = scores.get("ents_per_type", {})

    print(f"  {'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*54}")
    for label in PII_LABELS:
        if label in ents_per_type:
            s = ents_per_type[label]
            print(f"  {label:<20} {s['p']:>10.3f} {s['r']:>10.3f} {s['f']:>10.3f}")
        else:
            print(f"  {label:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    print(f"  {'-'*54}")
    print(f"  {'OVERALL':<20} {scores['ents_p']:>10.3f} {scores['ents_r']:>10.3f} {scores['ents_f']:>10.3f}")
    print()
    print("  P  = Precision  (of predicted entities, how many were correct)")
    print("  R  = Recall     (of true entities, how many did the model find)")
    print("  F1 = harmonic mean of P and R  ← main accuracy number")

if __name__ == "__main__":
    main()
