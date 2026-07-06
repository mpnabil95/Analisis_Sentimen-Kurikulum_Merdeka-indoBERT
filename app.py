import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocessing import preprocess_text

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis - Kurikulum Merdeka",
    page_icon="📊",
    layout="wide"
)

MODEL_PATH = "./model"  # <- SESUAIKAN DENGAN FOLDER KAMU

LABEL_MAP = {
    0: "Positif",
    1: "Negatif",
    2: "Netral"
}

CSV_PATH = "data/hasil_prediksi.csv"
os.makedirs("data", exist_ok=True)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        use_safetensors=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# PREPROCESS + PREDICT
# =========================
def predict_sentiment(text: str):
    text_clean = preprocess_text(text)

    inputs = tokenizer(
        text_clean,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()
    confidence = float(probs[0][pred_id])

    return {
        "text_original": text,
        "text_clean": text_clean,
        "label": LABEL_MAP[pred_id],
        "confidence": confidence
    }

# =========================
# SAVE TO CSV
# =========================
def save_to_csv(result: dict):
    df_new = pd.DataFrame([{
        "text": result["text_original"],
        "text_clean": result["text_clean"],
        "label": result["label"],
        "confidence": result["confidence"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(CSV_PATH):
        df_old = pd.read_csv(CSV_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(CSV_PATH, index=False)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    ["🏠 Dashboard", "🧠 Analisis Sentimen", "📂 Riwayat Data"]
)

st.sidebar.markdown("---")
st.sidebar.info("IndoBERT Sentiment Analysis - Kurikulum Merdeka")

# =========================
# DASHBOARD
# =========================
if menu == "🏠 Dashboard":
    st.title("📊 Dashboard Analisis Sentimen")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Analisis", len(df))
        col2.metric("Rata-rata Confidence", f"{df['confidence'].mean():.2f}")
        col3.metric("Dominan Sentimen", df['label'].mode()[0])

        st.markdown("### 📊 Distribusi Sentimen")

        fig = px.pie(
            df,
            names="label",
            title="Distribusi Sentimen"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📄 Data Terbaru")
        st.dataframe(df.tail(10), use_container_width=True)

    else:
        st.warning("Belum ada data analisis.")

# =========================
# INFERENCE PAGE
# =========================
elif menu == "🧠 Analisis Sentimen":
    st.title("🧠 Analisis Sentimen Kurikulum Merdeka")

    text_input = st.text_area("Masukkan teks opini:")

    if st.button("Analisis"):
        if text_input.strip() == "":
            st.error("Teks tidak boleh kosong")
        else:
            result = predict_sentiment(text_input)

            st.success(f"Sentimen: {result['label']}")
            st.info(f"Confidence: {result['confidence']:.2f}")

            st.markdown("### 🧹 Hasil Preprocessing")
            st.code(result["text_clean"])

            # SAVE HISTORY (SESSION)
            st.session_state.history.append(result)

            # SAVE TO CSV (PERMANENT)
            save_to_csv(result)

# =========================
# HISTORY PAGE
# =========================
elif menu == "📂 Riwayat Data":
    st.title("📂 Riwayat Analisis Sentimen")

    # SESSION HISTORY
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

    else:
        st.warning("Belum ada data di session.")

    # CSV HISTORY (GLOBAL)
    if os.path.exists(CSV_PATH):
        st.markdown("### 💾 Data Tersimpan (Global)")
        df_global = pd.read_csv(CSV_PATH)
        st.dataframe(df_global.tail(20), use_container_width=True)