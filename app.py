"""
app.py - Streamlit UI for the PII PDF Redactor.
"""
import io
from pathlib import Path

import streamlit as st
import pymupdf as fitz  # PyMuPDF

MODEL_DIR = "./model/pii_ner_model"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PII PDF Redactor",
    page_icon="🔒",
    layout="wide",
)

st.title("🔒 PII PDF Redactor")
st.caption("Detects and blacks-out personal information in PDF documents.")

# ── Model check ───────────────────────────────────────────────────────────────
if not Path(MODEL_DIR).exists():
    st.warning(
        "Trained model not found at `./model/pii_ner_model`.\n\n"
        "Run the training script first:\n```\npython train.py\n```",
        icon="⚠️",
    )
    st.stop()

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NER model …")
def get_model():
    from redactor import load_model
    return load_model()

nlp = get_model()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF file to get started.")
    st.stop()

pdf_bytes = uploaded.read()

# ── Redact button ─────────────────────────────────────────────────────────────
if st.button("🔴 Redact PDF", type="primary"):
    with st.spinner("Analysing and redacting PII …"):
        from redactor import redact_pdf
        redacted_bytes, entity_counts = redact_pdf(pdf_bytes, nlp)

    # ── Metrics bar ───────────────────────────────────────────────────────────
    total = sum(entity_counts.values())
    st.subheader(f"Entities detected: {total}")
    if entity_counts:
        cols = st.columns(len(entity_counts))
        for col, (label, count) in zip(cols, sorted(entity_counts.items())):
            col.metric(label, count)
    else:
        st.success("No PII detected in this document.")

    # ── Side-by-side preview (first page only) ────────────────────────────────
    st.subheader("Preview (page 1)")
    left, right = st.columns(2)

    def pdf_page_to_image(pdf_data: bytes, page_index: int = 0):
        doc  = fitz.open(stream=pdf_data, filetype="pdf")
        page = doc[page_index]
        pix  = page.get_pixmap(dpi=120)
        doc.close()
        return pix.tobytes("png")

    with left:
        st.markdown("**Original**")
        st.image(pdf_page_to_image(pdf_bytes), use_container_width=True)

    with right:
        st.markdown("**Redacted**")
        st.image(pdf_page_to_image(redacted_bytes), use_container_width=True)

    # ── Download button ───────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download Redacted PDF",
        data=redacted_bytes,
        file_name=f"redacted_{uploaded.name}",
        mime="application/pdf",
    )
