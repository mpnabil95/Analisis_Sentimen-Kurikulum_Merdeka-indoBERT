import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.preprocessing import preprocess_text

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Analysis - Kurikulum Merdeka",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
    }

    /* Hide default Streamlit noise */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e1b4b 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #f9fafb !important;
        font-weight: 600;
    }

    /* Cards */
    .hero-card {
        padding: 2.2rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #4338ca 100%);
        color: #ffffff;
        box-shadow: 0 24px 60px rgba(30, 27, 75, 0.26);
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.12;
        margin-bottom: 0.6rem;
        letter-spacing: -0.04em;
    }

    .hero-subtitle {
        font-size: 1.03rem;
        opacity: 0.92;
        max-width: 760px;
        line-height: 1.7;
    }

    .soft-card {
        padding: 1.25rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 20px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        min-height: 120px;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.9rem;
        color: #0f172a;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-help {
        font-size: 0.86rem;
        color: #64748b;
        margin-top: 0.35rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin: 1.4rem 0 0.8rem 0;
    }

    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 800;
        border: 1px solid rgba(255,255,255,0.3);
    }

    .badge-positive {
        background: rgba(22, 163, 74, 0.12);
        color: #15803d;
        border-color: rgba(22, 163, 74, 0.25);
    }

    .badge-negative {
        background: rgba(220, 38, 38, 0.12);
        color: #b91c1c;
        border-color: rgba(220, 38, 38, 0.25);
    }

    .badge-neutral {
        background: rgba(100, 116, 139, 0.14);
        color: #475569;
        border-color: rgba(100, 116, 139, 0.25);
    }

    .result-box {
        padding: 1.4rem;
        border-radius: 22px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
    }

    .empty-state {
        padding: 2rem;
        border-radius: 24px;
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        text-align: center;
        color: #475569;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .footer-note {
        color: #64748b;
        text-align: center;
        font-size: 0.86rem;
        margin-top: 2rem;
    }

    /* Button polish */
    div.stButton > button {
        border-radius: 14px;
        border: 0;
        background: linear-gradient(135deg, #4338ca, #6366f1);
        color: white;
        font-weight: 800;
        padding: 0.75rem 1rem;
        box-shadow: 0 10px 24px rgba(67, 56, 202, 0.22);
    }

    div.stButton > button:hover {
        border: 0;
        color: white;
        filter: brightness(1.04);
        transform: translateY(-1px);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
MODEL_PATH = "./model"

LABEL_MAP = {
    0: "Positif",
    1: "Negatif",
    2: "Netral"
}

SENTIMENT_COLOR_MAP = {
    "Positif": "#16a34a",
    "Negatif": "#dc2626",
    "Netral": "#64748b"
}

CSV_PATH = "data/hasil_prediksi.csv"
os.makedirs("data", exist_ok=True)

# =========================
# HELPER UI
# =========================
def render_hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def sentiment_badge(label: str):
    css_class = {
        "Positif": "badge-positive",
        "Negatif": "badge-negative",
        "Netral": "badge-neutral"
    }.get(label, "badge-neutral")

    return f'<span class="sentiment-badge {css_class}">{label}</span>'


def load_csv_history():
    if os.path.exists(CSV_PATH):
        try:
            return pd.read_csv(CSV_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def normalize_history_dataframe(df: pd.DataFrame):
    if df.empty:
        return df

    df = df.copy()

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource(show_spinner="Memuat model IndoBERT...")
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
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        "timestamp": result["timestamp"]
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

if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 📊 Sentiment App")
    st.caption("Analisis opini Kurikulum Merdeka berbasis IndoBERT")

    menu = st.radio(
        "Navigasi",
        ["🏠 Dashboard", "🧠 Analisis Sentimen", "📂 Riwayat Data"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Status Sistem")
    st.markdown(f"**Device:** `{device}`")
    st.markdown("**Model:** IndoBERT")
    st.markdown("**Output:** Positif, Negatif, Netral")

    st.markdown("---")
    st.info("Pastikan folder `model/` dan file `requirements.txt` sudah tersedia saat deploy.")

# =========================
# DASHBOARD
# =========================
if menu == "🏠 Dashboard":
    render_hero(
        "Dashboard Analisis Sentimen",
        "Pantau ringkasan hasil klasifikasi opini publik terhadap Kurikulum Merdeka melalui metrik, grafik distribusi, dan data prediksi terbaru."
    )

    df_global = normalize_history_dataframe(load_csv_history())

    if df_global.empty and len(st.session_state.history) > 0:
        df = normalize_history_dataframe(pd.DataFrame(st.session_state.history))
    else:
        df = df_global

    if not df.empty:
        total_data = len(df)
        avg_confidence = df["confidence"].mean() if "confidence" in df.columns else 0
        dominant_sentiment = df["label"].mode()[0] if "label" in df.columns and not df["label"].mode().empty else "-"

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric("Total Analisis", f"{total_data:,}", "Jumlah opini yang sudah diproses")
        with col2:
            render_metric("Rata-rata Confidence", f"{avg_confidence:.2%}", "Rata-rata keyakinan model")
        with col3:
            render_metric("Sentimen Dominan", dominant_sentiment, "Kategori yang paling sering muncul")

        st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns([1.1, 0.9])

        with chart_col1:
            sentiment_counts = df["label"].value_counts().reset_index()
            sentiment_counts.columns = ["label", "jumlah"]

            fig_bar = px.bar(
                sentiment_counts,
                x="label",
                y="jumlah",
                text="jumlah",
                color="label",
                color_discrete_map=SENTIMENT_COLOR_MAP,
                template="plotly_white"
            )
            fig_bar.update_layout(
                title=None,
                xaxis_title="Sentimen",
                yaxis_title="Jumlah",
                legend_title_text="Sentimen",
                height=420,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            fig_pie = px.pie(
                df,
                names="label",
                hole=0.55,
                color="label",
                color_discrete_map=SENTIMENT_COLOR_MAP,
                template="plotly_white"
            )
            fig_pie.update_layout(
                title=None,
                height=420,
                margin=dict(l=20, r=20, t=30, b=20),
                legend_title_text="Sentimen"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown('<div class="section-title">Data Prediksi Terbaru</div>', unsafe_allow_html=True)

        display_df = df.copy()
        if "timestamp" in display_df.columns:
            display_df = display_df.sort_values("timestamp", ascending=False)

        st.dataframe(
            display_df.tail(15) if "timestamp" not in display_df.columns else display_df.head(15),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.markdown(
            """
            <div class="empty-state">
                <h3>Belum ada data analisis</h3>
                <p>Masuk ke menu <b>Analisis Sentimen</b>, masukkan opini, lalu hasilnya akan tampil di dashboard ini.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================
# INFERENCE PAGE
# =========================
elif menu == "🧠 Analisis Sentimen":
    render_hero(
        "Analisis Sentimen Kurikulum Merdeka",
        "Masukkan opini, komentar, atau tanggapan masyarakat. Sistem akan melakukan preprocessing teks dan mengklasifikasikan sentimennya menggunakan model IndoBERT."
    )

    left_col, right_col = st.columns([1.15, 0.85])

    with left_col:
        st.markdown('<div class="section-title">Input Teks</div>', unsafe_allow_html=True)

        text_input = st.text_area(
            "Masukkan teks opini:",
            value=st.session_state.sample_text,
            height=190,
            placeholder="Contoh: Kurikulum Merdeka membantu siswa lebih aktif dalam proses pembelajaran...",
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns([0.35, 0.65])
        with col_btn1:
            analyze_clicked = st.button("Analisis Sekarang", use_container_width=True)
        with col_btn2:
            clear_clicked = st.button("Bersihkan Input", use_container_width=True)

        if clear_clicked:
            st.session_state.sample_text = ""
            st.rerun()

        st.markdown(
            """
            <div class="soft-card">
                <b>Tips input:</b>
                <div class="small-muted">
                    Gunakan kalimat opini yang jelas. Hindari input terlalu pendek seperti satu kata saja agar hasil prediksi lebih bermakna.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right_col:
        st.markdown('<div class="section-title">Contoh Opini</div>', unsafe_allow_html=True)

        sample_1 = "Kurikulum Merdeka membuat siswa lebih aktif dan guru lebih leluasa mengembangkan pembelajaran."
        sample_2 = "Pelaksanaan Kurikulum Merdeka masih membingungkan karena fasilitas sekolah belum merata."
        sample_3 = "Kurikulum Merdeka sudah diterapkan di beberapa sekolah dengan hasil yang beragam."

        if st.button("Contoh Positif", use_container_width=True):
            st.session_state.sample_text = sample_1
            st.rerun()

        if st.button("Contoh Negatif", use_container_width=True):
            st.session_state.sample_text = sample_2
            st.rerun()

        if st.button("Contoh Netral", use_container_width=True):
            st.session_state.sample_text = sample_3
            st.rerun()

        st.markdown(
            """
            <div class="soft-card">
                <b>Keluaran sistem:</b>
                <div class="small-muted">
                    Label sentimen, nilai confidence, dan hasil preprocessing teks.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if analyze_clicked:
        if text_input.strip() == "":
            st.error("Teks tidak boleh kosong.")
        else:
            with st.spinner("Sedang melakukan preprocessing dan prediksi sentimen..."):
                result = predict_sentiment(text_input)

            st.session_state.history.append(result)
            save_to_csv(result)

            st.markdown('<div class="section-title">Hasil Analisis</div>', unsafe_allow_html=True)

            result_col1, result_col2, result_col3 = st.columns([0.9, 0.8, 1.3])

            with result_col1:
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="metric-label">Sentimen</div>
                        {sentiment_badge(result["label"])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col2:
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{result["confidence"]:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(result["confidence"])

            with result_col3:
                st.markdown(
                    f"""
                    <div class="result-box">
                        <div class="metric-label">Waktu Analisis</div>
                        <div class="metric-value" style="font-size:1.05rem;">{result["timestamp"]}</div>
                        <div class="metric-help">Hasil otomatis disimpan ke CSV.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("Lihat hasil preprocessing", expanded=True):
                st.code(result["text_clean"], language="text")

            with st.expander("Lihat teks asli"):
                st.write(result["text_original"])

# =========================
# HISTORY PAGE
# =========================
elif menu == "📂 Riwayat Data":
    render_hero(
        "Riwayat Analisis Sentimen",
        "Lihat, filter, dan unduh hasil prediksi yang tersimpan selama penggunaan aplikasi."
    )

    tab1, tab2 = st.tabs(["Session Saat Ini", "Data Tersimpan CSV"])

    with tab1:
        if len(st.session_state.history) > 0:
            df_session = normalize_history_dataframe(pd.DataFrame(st.session_state.history))

            st.dataframe(
                df_session.sort_values("timestamp", ascending=False) if "timestamp" in df_session.columns else df_session,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="Download CSV Session",
                data=df_session.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_session.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <h3>Belum ada data di session ini</h3>
                    <p>Data session akan muncul setelah kamu melakukan analisis pada menu Analisis Sentimen.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        df_global = normalize_history_dataframe(load_csv_history())

        if not df_global.empty:
            filter_col1, filter_col2 = st.columns([0.7, 0.3])

            with filter_col1:
                selected_labels = st.multiselect(
                    "Filter sentimen",
                    options=sorted(df_global["label"].dropna().unique().tolist()),
                    default=sorted(df_global["label"].dropna().unique().tolist())
                )

            with filter_col2:
                limit_rows = st.number_input(
                    "Jumlah baris tampil",
                    min_value=5,
                    max_value=500,
                    value=50,
                    step=5
                )

            filtered_df = df_global[df_global["label"].isin(selected_labels)] if selected_labels else df_global

            if "timestamp" in filtered_df.columns:
                filtered_df = filtered_df.sort_values("timestamp", ascending=False)

            st.dataframe(
                filtered_df.head(limit_rows),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="Download Semua Data CSV",
                data=df_global.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_global.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.markdown(
                """
                <div class="empty-state">
                    <h3>Belum ada file CSV tersimpan</h3>
                    <p>File akan dibuat otomatis setelah analisis pertama dilakukan.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown(
    """
    <div class="footer-note">
        IndoBERT Sentiment Analysis App · Kurikulum Merdeka
    </div>
    """,
    unsafe_allow_html=True
)
