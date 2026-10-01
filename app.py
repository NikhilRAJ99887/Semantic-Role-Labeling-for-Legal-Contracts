import re
import streamlit as st
from pypdf import PdfReader
from docx import Document
from transformers import pipeline


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide"
)


# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = "outputs/legal-bert-ner"


# =====================================================
# LOAD BERT MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = pipeline(
        "token-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        aggregation_strategy="simple"
    )

    return model


# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =====================================================
# DOCX TEXT EXTRACTION
# =====================================================

def extract_docx(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text += paragraph.text + "\n"

    return text


# =====================================================
# TXT EXTRACTION
# =====================================================

def extract_txt(file):

    return file.read().decode("utf-8")


# =====================================================
# SEMANTIC ROLE MAPPING
# =====================================================

WHO_LABELS = [
    "PERSON",
    "PARTY",
    "ROLE"
]

WHAT_LABELS = [
    "ACTION",
    "DELIVERABLE",
    "PAYMENT",
    "AMOUNT",
    "CONDITION",
    "NOTICE"
]

WHEN_LABELS = [
    "DATE",
    "DEADLINE",
    "DURATION"
]


def get_category(label):

    # Remove BIO prefix

    label = label.replace("B-", "")
    label = label.replace("I-", "")

    if label in WHO_LABELS:

        return "WHO"

    elif label in WHAT_LABELS:

        return "WHAT"

    elif label in WHEN_LABELS:

        return "WHEN"

    else:

        return "OTHER"

def merge_bert_entities(results):
    """
    Merge BERT WordPiece tokens into normal readable words.
    Example:
        ra + ##hul sharma
    becomes:
        Rahul Sharma
    """

    if not results:
        return []

    merged = []

    for entity in results:

        results = model(chunk)

        merged_results = merge_bert_entities(results)

        for entity in merged_results:

            word = entity["word"]
            label = entity["label"]
            category = entity["category"]

            detailed_entities.append({
                "word": word,
                "label": label,
                "category": category
            })

            # WHO
            if category == "WHO":

                if word not in who:
                    who.append(word)

            # WHAT
            elif category == "WHAT":

                if word not in what:
                    what.append(word)

            # WHEN
            elif category == "WHEN":

                if word not in when:
                    when.append(word)

        # Check whether this belongs to the previous entity
        if merged:

            previous = merged[-1]

            same_category = (
                previous["category"] == category
            )

            # If BERT says the next token starts immediately
            # after the previous token, join them.
            adjacent = False

            if (
                previous["end"] is not None
                and current["start"] is not None
            ):
                adjacent = (
                    current["start"] <= previous["end"] + 1
                )

            # Also join WordPiece-style continuation
            if same_category and adjacent:

                previous["word"] = (
                    previous["word"] + " " + current["word"]
                )

                previous["end"] = current["end"]

                continue

        merged.append(current)

    return merged
# =====================================================
# ANALYZE TEXT
# =====================================================
def analyze_text(text):

    model = load_model()

    words = text.split()

    chunk_size = 150

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)


    who = []
    what = []
    when = []

    detailed_entities = []


    # Run BERT
    for chunk in chunks:

        results = model(chunk)

        for entity in results:

            word = entity.get("word", "")

            # Support both output formats
            label = entity.get(
                "entity_group",
                entity.get("label", "")
            )

            category = get_category(label)


            # Save entity
            detailed_entities.append({
                "word": word,
                "label": label,
                "category": category
            })


            # WHO
            if category == "WHO":

                if word not in who:
                    who.append(word)


            # WHAT
            elif category == "WHAT":

                if word not in what:
                    what.append(word)


            # WHEN
            elif category == "WHEN":

                if word not in when:
                    when.append(word)


    return (
        who,
        what,
        when,
        detailed_entities
    )


# =====================================================
# MAIN PAGE
# =====================================================

st.title("⚖️ Legal Contract Analyzer")

st.write(
    "BERT-based Semantic Role Labeling "
    "for Freelance Software Development Contracts"
)


st.markdown("---")


# =====================================================
# INPUT METHOD
# =====================================================

st.subheader("Choose Input Method")

input_method = st.radio(
    "How do you want to provide the contract?",
    [
        "📄 Upload Contract",
        "⌨️ Type / Paste Contract"
    ],
    horizontal=True
)


text = ""


# =====================================================
# OPTION 1: FILE UPLOAD
# =====================================================

if input_method == "📄 Upload Contract":

    uploaded_file = st.file_uploader(
        "Upload your contract",
        type=["pdf", "docx", "txt"]
    )


    if uploaded_file is not None:

        file_name = uploaded_file.name.lower()


        if file_name.endswith(".pdf"):

            text = extract_pdf(
                uploaded_file
            )


        elif file_name.endswith(".docx"):

            text = extract_docx(
                uploaded_file
            )


        elif file_name.endswith(".txt"):

            text = extract_txt(
                uploaded_file
            )


        if text.strip():

            st.success(
                f"Successfully extracted text from "
                f"{uploaded_file.name}"
            )


            with st.expander(
                "📖 View Extracted Contract"
            ):

                st.text_area(
                    "Extracted text",
                    text,
                    height=300
                )


        else:

            st.error(
                "No readable text was found "
                "in the uploaded file."
            )


# =====================================================
# OPTION 2: TYPE CONTRACT
# =====================================================

else:

    st.subheader(
        "Enter Contract Text"
    )


    text = st.text_area(
        "Type or paste your contract here:",
        height=300,
        placeholder=(
            "Example:\n\n"
            "ABC Technologies appoints Rahul Sharma "
            "as a software developer. Rahul shall "
            "develop and deliver the mobile application "
            "by 30 November 2026. The client shall pay "
            "Rahul ₹50,000 upon completion."
        )
    )


# =====================================================
# ANALYZE BUTTON
# =====================================================

st.markdown("---")


if st.button(
    "🔍 Analyze Contract",
    type="primary",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "Please upload a contract or enter "
            "some contract text first."
        )

        st.stop()


    with st.spinner(
        "BERT is analyzing the contract..."
    ):

        try:

            who, what, when, entities = analyze_text(
                text
            )


        except Exception as e:

            st.error(
                f"Error while analyzing contract: {e}"
            )

            st.stop()


    # =================================================
    # RESULTS
    # =================================================

    st.markdown("---")

    st.header(
        "📋 Extracted Contract Information"
    )


    col1, col2, col3 = st.columns(3)


    # =================================================
    # WHO
    # =================================================

    with col1:

        st.subheader("👤 WHO")

        if who:

            for item in who:

                st.success(item)

        else:

            st.info(
                "No WHO information found."
            )


    # =================================================
    # WHAT
    # =================================================

    with col2:

        st.subheader("📌 WHAT")

        if what:

            for item in what:

                st.success(item)

        else:

            st.info(
                "No WHAT information found."
            )


    # =================================================
    # WHEN
    # =================================================

    with col3:

        st.subheader("📅 WHEN")

        if when:

            for item in when:

                st.success(item)

        else:

            st.info(
                "No WHEN information found."
            )


    # =================================================
    # DETAILED NER RESULTS
    # =================================================

    st.markdown("---")

    st.subheader(
        "🔎 Detailed BERT NER Results"
    )


    if entities:

        for entity in entities:

            st.write(
                f"**{entity['word']}** "
                f"→ `{entity['label']}` "
                f"→ **{entity['category']}**"
            )

    else:

        st.info(
            "No entities detected."
        )