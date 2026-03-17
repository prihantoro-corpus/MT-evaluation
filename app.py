import streamlit as st
import pandas as pd
from langdetect import detect
import sacrebleu
import matplotlib.pyplot as plt
import requests
import re
from io import BytesIO
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 MT Evaluation Tool (Advanced Multilingual + Japanese NLP Support)")

# ======================
# WARNING
# ======================
st.warning(
    "⚠️ Experimental Feature: Linguistic error analysis uses heuristic and tokenizer-based methods. Results may vary depending on tokenizer availability (SudachiPy/MeCab)."
)

# ======================
# TOKENIZER SETUP
# ======================
tokenizer_mode = "char"

# Try SudachiPy
try:
    from sudachipy import tokenizer as sudachi_tokenizer
    from sudachipy import dictionary

    sudachi_obj = dictionary.Dictionary().create()
    tokenizer_mode = "sudachi"
except:
    # Try MeCab (via fugashi)
    try:
        from fugashi import Tagger
        mecab = Tagger()
        tokenizer_mode = "mecab"
    except:
        tokenizer_mode = "char"

def tokenize(text, lang):
    if lang in ["ja", "zh-cn", "zh-tw", "ko"]:

        if tokenizer_mode == "sudachi":
            return [m.surface() for m in sudachi_obj.tokenize(text)]

        elif tokenizer_mode == "mecab":
            return [word.surface for word in mecab(text)]

        else:
            return list(text.strip())

    else:
        return text.split()

# ======================
# ERROR ANALYSIS V4
# ======================
def error_analysis_v4(src, mt, ref, tgt_lang="en"):

    try:
        errors = []

        mt_tokens = tokenize(mt, tgt_lang)
        ref_tokens = tokenize(ref, tgt_lang)
        src_tokens = tokenize(src, tgt_lang)

        # ===== CONTENT =====
        if len(mt_tokens) < 0.7 * len(ref_tokens):
            errors.append("Omission")

        if len(mt_tokens) > 1.3 * len(ref_tokens):
            errors.append("Addition")

        overlap = len(set(mt_tokens) & set(ref_tokens)) / max(len(set(ref_tokens)), 1)

        if overlap < 0.3:
            errors.append("Mistranslation")

        # ===== LEXICAL =====
        if overlap < 0.5 and len(mt_tokens) > 0.8 * len(ref_tokens):
            errors.append("Lexical Choice")

        shared = set(src_tokens) & set(mt_tokens)
        if len(shared) > 2:
            errors.append("Untranslated Segment")

        # ===== FORMAL =====
        if re.findall(r"\d+", mt) != re.findall(r"\d+", ref):
            errors.append("Number Mismatch")

        if mt.count("。") != ref.count("。") and mt.count(".") != ref.count("."):
            errors.append("Punctuation Error")

        # ===== GRAMMAR (ONLY ENGLISH) =====
        if tgt_lang == "en":

            if re.search(r"\b(he|she|it)\s+go\b", mt):
                errors.append("Agreement Error")

            if "very" in mt and not re.search(r"\b(is|are|was|were)\b", mt):
                errors.append("Missing Copula")

            if re.search(r"\b(am|is|are)\s+\w+\b", mt):
                if "ing" not in mt:
                    errors.append("Incorrect Verb Form")

            if "yesterday" in ref and "go" in mt:
                errors.append("Tense Error")

        return ", ".join(sorted(set(errors))) if errors else "OK"

    except:
        return "Unknown"

# ======================
# TABS
# ======================
tab1, tab2 = st.tabs(["📂 Upload & Settings", "📈 Results"])

# ======================
# TAB 1
# ======================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

    with col2:
        mode = st.radio("Mode", ["Lite (default)", "Advanced (API optional)"])
        api_key = None
        if mode == "Advanced (API optional)":
            api_key = st.text_input("HuggingFace API Key", type="password")

# ======================
# TAB 2
# ======================
with tab2:

    if uploaded_file:

        df = pd.read_excel(uploaded_file)

        if df.shape[1] < 3:
            st.error("File must have at least 3 columns: source, mt, ref")
            st.stop()

        df = df.iloc[:, :3]
        df.columns = ["source", "mt", "ref"]

        # ======================
        # LANGUAGE DETECTION
        # ======================
        try:
            src_lang = detect(str(df["source"].iloc[0]))
            tgt_lang = detect(str(df["ref"].iloc[0]))
        except:
            src_lang, tgt_lang = "unknown", "unknown"

        st.write(f"Detected: {src_lang} → {tgt_lang}")
        st.write(f"Tokenizer mode: {tokenizer_mode}")

        # ======================
        # CLEAN
        # ======================
        def clean(x):
            return str(x).strip().lower()

        df["mt_clean"] = df["mt"].apply(clean)
        df["ref_clean"] = df["ref"].apply(clean)

        if st.button("🚀 Run Evaluation"):

            # ======================
            # METRICS
            # ======================
            bleu = sacrebleu.corpus_bleu(df["mt_clean"], [df["ref_clean"]])
            chrf = sacrebleu.corpus_chrf(df["mt_clean"], [df["ref_clean"]], word_order=2)

            st.subheader("📊 Metrics")

            colA, colB = st.columns(2)
            colA.metric("BLEU", f"{bleu.score:.2f}")
            colB.metric("chrF++", f"{chrf.score:.2f}")

            # ======================
            # SENTENCE LEVEL
            # ======================
            df["bleu_sent"] = [
                sacrebleu.sentence_bleu(m, [r]).score for m, r in zip(df["mt_clean"], df["ref_clean"])
            ]

            df["chrf_sent"] = [
                sacrebleu.sentence_chrf(m, [r], word_order=2).score for m, r in zip(df["mt_clean"], df["ref_clean"])
            ]

            # ======================
            # ERROR ANALYSIS
            # ======================
            df["error_type"] = [
                error_analysis_v4(s, m, r, tgt_lang)
                for s, m, r in zip(df["source"], df["mt"], df["ref"])
            ]

            # ======================
            # VISUALIZATION
            # ======================
            st.subheader("📈 chrF++ Distribution")
            fig, ax = plt.subplots()
            ax.hist(df["chrf_sent"], bins=20)
            st.pyplot(fig)

            st.subheader("⚠️ Error Distribution")
            st.bar_chart(df["error_type"].value_counts())

            # ======================
            # TABLE
            # ======================
            st.dataframe(df)

            # ======================
            # EXPORT
            # ======================
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)

            st.download_button(
                "⬇️ Download Excel",
                data=excel_buffer.getvalue(),
                file_name="mt_results.xlsx"
            )
