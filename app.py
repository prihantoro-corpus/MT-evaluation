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
st.title("📊 MT Evaluation Tool (Stable Multilingual Version)")

# ======================
# WARNING
# ======================
st.warning(
    "⚠️ Experimental Feature: Linguistic error analysis is heuristic-based. "
    "For Japanese/Chinese/Korean, character-based fallback is used if tokenizer is unavailable."
)

# ======================
# TOKENIZER SETUP (SAFE)
# ======================
tokenizer_mode = "char"

try:
    from sudachipy import dictionary
    sudachi_obj = dictionary.Dictionary().create()
    tokenizer_mode = "sudachi"
except:
    tokenizer_mode = "char"

def is_cjk(lang):
    return lang in ["ja", "zh-cn", "zh-tw", "ko"]

def tokenize(text, lang):
    if is_cjk(lang):
        if tokenizer_mode == "sudachi":
            try:
                return [m.surface() for m in sudachi_obj.tokenize(text)]
            except:
                return list(text.strip())
        else:
            return list(text.strip())
    else:
        return text.split()

# ======================
# ERROR ANALYSIS
# ======================
def error_analysis(src, mt, ref, tgt_lang):

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

        # ===== GRAMMAR (ENGLISH ONLY)
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
# UI TABS
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
        mode = st.radio("Mode", ["Lite", "Advanced (COMET optional)"])
        api_key = None
        if mode == "Advanced (COMET optional)":
            api_key = st.text_input("HuggingFace API Key", type="password")

# ======================
# TAB 2
# ======================
with tab2:

    if uploaded_file:

        try:
            df = pd.read_excel(uploaded_file)
        except:
            st.error("Failed to read Excel file.")
            st.stop()

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

        if is_cjk(tgt_lang):
            st.info("🈶 CJK detected → using character/token hybrid strategy")

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
            try:
                bleu = sacrebleu.corpus_bleu(df["mt_clean"], [df["ref_clean"]])
                chrf = sacrebleu.corpus_chrf(
                    df["mt_clean"],
                    [df["ref_clean"]],
                    word_order=2
                )

                st.subheader("📊 Metrics")

                colA, colB = st.columns(2)
                colA.metric("BLEU", f"{bleu.score:.2f}")
                colB.metric("chrF++", f"{chrf.score:.2f}")

                if is_cjk(tgt_lang):
                    st.caption("👉 chrF++ is more reliable for CJK languages.")

            except:
                st.warning("Metric calculation failed")

            # ======================
            # SENTENCE LEVEL
            # ======================
            def safe_bleu(mt, ref):
                try:
                    return sacrebleu.sentence_bleu(mt, [ref]).score
                except:
                    return np.nan

            def safe_chrf(mt, ref):
                try:
                    return sacrebleu.sentence_chrf(mt, [ref], word_order=2).score
                except:
                    return np.nan

            df["bleu_sent"] = [safe_bleu(m, r) for m, r in zip(df["mt_clean"], df["ref_clean"])]
            df["chrf_sent"] = [safe_chrf(m, r) for m, r in zip(df["mt_clean"], df["ref_clean"])]

            # ======================
            # COMET (OPTIONAL)
            # ======================
            if api_key:
                st.subheader("🧠 COMET (Optional)")
                try:
                    url = "https://api-inference.huggingface.co/models/Unbabel/wmt22-comet-da"
                    headers = {"Authorization": f"Bearer {api_key}"}

                    data = {
                        "inputs": [
                            {"src": s, "mt": m, "ref": r}
                            for s, m, r in zip(df["source"], df["mt"], df["ref"])
                        ]
                    }

                    response = requests.post(url, headers=headers, json=data)

                    if response.status_code == 200:
                        result = response.json()
                        df["comet"] = [x["score"] for x in result]
                        st.success("COMET success")
                    else:
                        st.warning("COMET API failed")

                except Exception as e:
                    st.warning(f"COMET error: {e}")

            # ======================
            # QUALITY LABEL
            # ======================
            def interpret(score):
                if pd.isna(score):
                    return "Unrated"
                elif score > 65:
                    return "Good"
                elif score > 40:
                    return "Fair"
                else:
                    return "Poor"

            df["quality"] = df["bleu_sent"].apply(interpret)

            # ======================
            # ERROR ANALYSIS
            # ======================
            df["error_type"] = [
                error_analysis(s, m, r, tgt_lang)
                for s, m, r in zip(df["source"], df["mt"], df["ref"])
            ]

            # ======================
            # VISUALIZATION
            # ======================
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots()
                ax.hist(df["chrf_sent"].dropna(), bins=20)
                ax.set_title("chrF++ Distribution")
                st.pyplot(fig)

            with col2:
                st.bar_chart(df["quality"].value_counts())

            st.subheader("📈 BLEU vs chrF++")
            compare_df = pd.DataFrame({
                "BLEU": df["bleu_sent"],
                "chrF++": df["chrf_sent"]
            })
            st.line_chart(compare_df)

            st.subheader("⚠️ Error Distribution")
            st.bar_chart(df["error_type"].value_counts())

            # ======================
            # TABLE
            # ======================
            st.subheader("📋 Detailed Results")
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

            def generate_txt(df):
                text = "MT Evaluation Report\n\n"
                for i, row in df.iterrows():
                    text += f"{i+1}. {row['mt']} | {row['quality']} | {row['error_type']}\n"
                return text

            st.download_button(
                "⬇️ Download TXT",
                data=generate_txt(df),
                file_name="mt_report.txt"
            )
