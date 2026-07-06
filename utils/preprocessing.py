# utils/preprocessing.py

import re

# =========================
# SLANG NORMALIZATION DICTIONARY
# =========================
SLANG_DICT = {
    "yg": "yang", "dgn": "dengan", "tdk": "tidak", "krn": "karena",
    "bgt": "banget", "kurmer": "kurikulum merdeka", "tp": "tapi",
    "tpi": "tapi", "th": "tahun", "ttg": "tentang", "dg": "dengan",
    "gak": "tidak", "ga": "tidak", "udh": "sudah", "sy": "saya",
    "jd": "jadi", "kalo": "kalau", "utk": "untuk", "dr": "dari",
    "pndidikan": "pendidikan", "ngajar": "mengajar", "trs": "terus",
    "sbg": "sebagai", "sj": "saja", "ngga": "tidak", "sdh": "sudah",
    "pd": "pada", "sprt": "seperti", "dlm": "dalam", "bs":"bisa",
    "ttp": "tetap", "bnyk": "banyak", "msh": "masih", "skrg": "sekarang",
    "gk": "tidak", "kpd": "kepada", "jg": "juga", "hrs": "harus",
    "dpt": "dapat", "lbh": "lebih", "kl": "kalau", "bljr": "belaajar",
    "blm": "belum", "bgus": "bagus", "ad": "ada", "kyk": "seperti",
    "brp": "berapa", "knp": "kenapa", "dri": "dari", "jga": "juga",
    "thn": "tahun", "g": "tidak", "tsb": "tersebut", "bkn": "bukan",
    "bgs": "bagus", "gurgem": "guru gembul", "trus": "terus",
    "kurmer": "kurikulum merdeka", "kumer": "kurikulum merdeka",
    "kepsek": "kepala sekolah", "pmm": "platform merdeka mengajar",
    "kalo": "kalau", "klo": "kalau", "gk": "tidak", "tp": "tapi",
    "gmn": "bagaimana", "ribet": "rumit", "org": "orang", "jgn": "jangan",
    "gt": "gitu", "kt": "kata", "aj": "aja", "gw": "saya"
}

# =========================
# CLEAN TEXT FUNCTION
# =========================
def clean_text(text: str) -> str:
    """
    Cleaning teks untuk NLP IndoBERT sentiment analysis
    """
    text = str(text).lower()

    # remove mention
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    # remove hashtag
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)

    # remove URL
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================
# NORMALIZE SLANG
# =========================
def normalize_slang(text: str) -> str:
    """
    Mengubah kata tidak baku menjadi baku
    """
    words = text.split()
    words = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(words)


# =========================
# FULL PIPELINE PREPROCESSING
# =========================
def preprocess_text(text: str) -> str:
    """
    Pipeline lengkap preprocessing:
    clean -> normalize slang
    """
    text = clean_text(text)
    text = normalize_slang(text)
    return text


# =========================
# OPTIONAL: BATCH PROCESS
# =========================
def preprocess_batch(texts):
    """
    Untuk dataset / dataframe
    """
    return [preprocess_text(t) for t in texts]