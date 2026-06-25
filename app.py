import json
import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from shade_card import create_shade_card_png
from ui_helpers import (
    normalize_tips,
    render_tone_box,
    render_confidence_bars,
    render_product_cards,
    render_match_summary,
    render_faq,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MatchMyShade",
    page_icon="M",
    layout="centered",
)

# ============================================================
# CUSTOM CSS
# ============================================================

# Load external CSS file 
CSS_PATH = Path("styles.css")
if CSS_PATH.exists():
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning("File styles.css tidak ditemukan. Tampilan aplikasi akan menggunakan style default Streamlit.")


# ============================================================
# PATHS — lokasi model, data, dan aset
# ============================================================

# MODEL_DIR khusus untuk file model (.keras) beserta metadata-nya.
# DATA_DIR khusus untuk dataset dan konfigurasi recommendation system.
# Dipisah agar struktur folder semantically correct: model != data.
MODEL_DIR         = Path("models")
DATA_DIR          = Path("data")
ASSET_DIR         = Path("assets")
HEADER_IMAGE_PATH = ASSET_DIR / "matchmyshade_header.png"

# Model files
SKIN_MODEL_PATH   = MODEL_DIR / "final_mobilenetv2_skin_tone.keras"
SKIN_META_PATH    = MODEL_DIR / "model_metadata.json"
FACE_MODEL_PATH   = MODEL_DIR / "face_validator_model.keras"
FACE_META_PATH    = MODEL_DIR / "face_validator_metadata.json"

# Data files
FOUNDATION_CSV_PATH  = DATA_DIR / "foundation_products_cleaned.csv"
TARGET_L_PATH        = DATA_DIR / "target_lightness.json"
LIGHTNESS_RANGE_PATH = DATA_DIR / "lightness_range.json"
SKIN_TYPE_RULES_PATH = DATA_DIR / "skin_type_rules.json"

# Semua file yang wajib ada sebelum app bisa jalan
REQUIRED_FILES = [
    SKIN_MODEL_PATH, SKIN_META_PATH,
    FACE_MODEL_PATH, FACE_META_PATH,
    FOUNDATION_CSV_PATH, TARGET_L_PATH,
    LIGHTNESS_RANGE_PATH, SKIN_TYPE_RULES_PATH,
]

# ============================================================
# CONSTANTS
# ============================================================

IMG_SIZE             = 224    # ukuran input model (px)
CONFIDENCE_THRESHOLD = 0.60   # batas minimum confidence sebelum dianggap ambigu
FACE_CROP_MARGIN     = 0.35   # padding di sekitar face crop (rasio)
DEFAULT_TOP_N        = 6      # jumlah rekomendasi produk default

# Warna representasi tiap skin tone untuk UI.
SKIN_TONE_COLORS = {
    "Fair"  : "#F3D7C2",
    "Medium": "#C98F68",
    "Tan"   : "#8D5524",
    "Deep"  : "#4B2E1E",
}
ALL_SKIN_TONES = ["Fair", "Medium", "Tan", "Deep"]

# Label skin type untuk dropdown (bilingual)
SKIN_TYPE_LABELS = {
    "oily"       : "Oily (Berminyak)",
    "dry"        : "Dry (Kering)",
    "combination": "Combination (Kombinasi)",
    "normal"     : "Normal",
    "sensitive"  : "Sensitive (Sensitif)",
}

# Label pendek untuk summary card
SKIN_TYPE_LABELS_SHORT = {
    "oily"       : "Oily",
    "dry"        : "Dry",
    "combination": "Combination",
    "normal"     : "Normal",
    "sensitive"  : "Sensitive",
}

# ============================================================
# LOAD ASSETS — model dan data di-cache agar tidak reload tiap interaksi
# ============================================================

@st.cache_resource(show_spinner="Loading skin tone model...")
def load_skin_model():
    return tf.keras.models.load_model(SKIN_MODEL_PATH, safe_mode=False)

@st.cache_resource(show_spinner="Loading face validator...")
def load_face_model():
    return tf.keras.models.load_model(FACE_MODEL_PATH, safe_mode=False)

@st.cache_data(show_spinner=False)
def load_assets():
    # Load semua JSON config
    with open(SKIN_META_PATH) as f:
        skin_meta = json.load(f)
    with open(FACE_META_PATH) as f:
        face_meta = json.load(f)
    with open(TARGET_L_PATH) as f:
        target_lightness = json.load(f)
    with open(LIGHTNESS_RANGE_PATH) as f:
        lightness_range = {k: tuple(v) for k, v in json.load(f).items()}
    with open(SKIN_TYPE_RULES_PATH) as f:
        skin_type_rules = json.load(f)

    # Load dan bersihkan dataset foundation
    df = pd.read_csv(FOUNDATION_CSV_PATH)
    df["hex"] = df["hex"].astype(str).str.strip()
    df["hex"] = df["hex"].apply(lambda x: x if x.startswith("#") else f"#{x}")
    df["lightness"] = pd.to_numeric(df["lightness"], errors="coerce")
    if df["lightness"].max() > 1:
        df["lightness"] = df["lightness"] / 100  # normalisasi ke 0–1
    df = df.dropna(subset=["hex", "lightness"])

    return skin_meta, face_meta, target_lightness, lightness_range, skin_type_rules, df

# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def preprocess(pil_img):
    """Resize dan expand dims gambar untuk input model."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(img, dtype="float32"), axis=0)

def validate_face(pil_img, face_model, face_meta):
    """Validasi apakah gambar mengandung wajah manusia.

    Output sigmoid merepresentasikan probabilitas terhadap label 1.
    Karena urutan class_names dapat berbeda, keputusan wajah/non-wajah
    dibaca dari metadata human_label agar tidak tertukar.
    """
    prob      = float(face_model.predict(preprocess(pil_img), verbose=0)[0][0])
    threshold = float(face_meta.get("threshold", 0.5))

    class_names = face_meta.get("class_names", [])
    human_label = face_meta.get("human_label", None)

    # Fallback untuk metadata lama.
    if human_label is None:
        human_class = face_meta.get("human_class", "human_faces")
        if class_names and human_class in class_names:
            human_label = class_names.index(human_class)
        else:
            positive_cls = face_meta.get("positive_class", face_meta.get("label_1_class", "non_human_faces"))
            human_label = 0 if positive_cls == "non_human_faces" else 1

    human_label = int(human_label)

    # Skor >= threshold berarti prediksi label 1, selain itu label 0.
    pred_label = int(prob >= threshold)
    is_face = pred_label == human_label

    # Confidence terhadap keputusan akhir.
    confidence = prob if human_label == 1 else (1.0 - prob)

    return {
        "is_face"    : bool(is_face),
        "confidence" : round(float(confidence), 4),
        "raw_score"  : round(float(prob), 4),
        "pred_label" : pred_label,
        "human_label": human_label,
    }

def crop_face(pil_img, margin=FACE_CROP_MARGIN):
    """Crop area wajah menggunakan Haar Cascade dari OpenCV."""
    img_np  = np.array(pil_img.convert("RGB"))
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return pil_img, False  # fallback ke gambar asli jika wajah tidak ditemukan
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    H, W = img_np.shape[:2]
    pad_x = int(w * margin); pad_y = int(h * margin)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x); y2 = min(H, y + h + pad_y)
    return Image.fromarray(img_np[y1:y2, x1:x2]), True

def predict_skin_tone(pil_img, skin_model, skin_meta):
    """Prediksi skin tone menggunakan MobileNetV2."""
    preds       = skin_model.predict(preprocess(pil_img), verbose=0)[0]
    class_names = skin_meta.get("mapped_class_names", [])
    disp_labels = skin_meta.get("display_labels", {})
    labels      = class_names if class_names else [disp_labels.get(str(i), str(i)) for i in range(len(preds))]
    pred_idx    = int(np.argmax(preds))
    prob_dict   = {labels[i]: round(float(preds[i]), 4) for i in range(len(preds))}
    return {
        "predicted_skin_tone": labels[pred_idx],
        "confidence"         : round(float(preds[pred_idx]), 4),
        "probabilities"      : prob_dict,
    }

def check_ambiguity(probabilities, threshold=CONFIDENCE_THRESHOLD):
    """Cek apakah prediksi ambigu (confidence di bawah threshold)."""
    top2        = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:2]
    t1_l, t1_c = top2[0]
    t2_l, t2_c = top2[1] if len(top2) > 1 else (None, 0.0)
    return {
        "is_ambiguous": t1_c < threshold,
        "top1": {"label": t1_l, "confidence": t1_c},
        "top2": {"label": t2_l, "confidence": t2_c},
    }

def calculate_dynamic_target(probabilities, target_lightness):
    """Hitung target lightness dinamis sebagai weighted average dari semua confidence."""
    total = 0.0; val = 0.0
    for tone, prob in probabilities.items():
        if tone in target_lightness:
            val   += float(target_lightness[tone]) * float(prob)
            total += float(prob)
    return val / total if total > 0 else None

def recommend_foundation(skin_tone, skin_type, df, target_lightness,
                         lightness_range, skin_type_rules, top_n, target_override=None):
    """Cari shade foundation paling mendekati target lightness berdasarkan skin tone final."""
    skin_tone = str(skin_tone).title()
    skin_type = str(skin_type).lower()

    target = float(target_override) if target_override is not None else float(target_lightness[skin_tone])
    rule = skin_type_rules.get(skin_type, skin_type_rules.get("normal", {}))

    # Pool produk diambil dari kategori skin tone hasil mapping
    if "skin_tone" in df.columns:
        pool = df[df["skin_tone"] == skin_tone].copy()
    else:
        l_min, l_max = lightness_range[skin_tone]
        if skin_tone == "Fair":
            pool = df[(df["lightness"] >= l_min) & (df["lightness"] <= l_max)].copy()
        else:
            pool = df[(df["lightness"] >= l_min) & (df["lightness"] < l_max)].copy()

    if pool.empty:
        pool = df.copy()

    pool["lightness_distance"] = (pool["lightness"] - target).abs()
    top = pool.sort_values(
        ["lightness_distance", "brand", "product", "hex"]
    ).head(top_n).reset_index(drop=True)

    base_cols = ["brand", "product", "hex", "lightness", "lightness_distance"]
    optional_cols = ["name", "specific", "url"]
    avail_cols = base_cols + [c for c in optional_cols if c in top.columns]

    products = (
        top[avail_cols]
        .rename(columns={"name": "shade_name", "specific": "shade_code"})
        .to_dict(orient="records")
    )

    return {
        "skin_tone": skin_tone,
        "skin_type": skin_type,
        "target_lightness": target,
        "suggested_finish": rule.get("suggested_finish", rule.get("finish", "")),
        "suggested_coverage": rule.get("suggested_coverage", rule.get("coverage", "")),
        "tips": normalize_tips(rule.get("tips", "")),
        "recommended_products": products,
    }

def run_pipeline(pil_img, skin_type, top_n, face_model, face_meta,
                 skin_model, skin_meta, df, target_lightness,
                 lightness_range, skin_type_rules, manual_tone=None):
    """
    Pipeline utama: validasi wajah → crop → prediksi skin tone → rekomendasi.
    Mengembalikan dict dengan status 'accepted' atau 'rejected'.
    """
    # Step 1: validasi wajah pada gambar asli
    face_orig = validate_face(pil_img, face_model, face_meta)
    if not face_orig["is_face"]:
        return {
            "status" : "rejected",
            "stage"  : "face_validation_original",
            "message": "Wajah tidak terdeteksi. Pastikan foto menampilkan wajah dengan jelas dan pencahayaan yang cukup.",
            "face_orig": face_orig,
        }

    # Step 2: crop wajah, fallback ke gambar asli jika gagal
    cropped_img, crop_success = crop_face(pil_img)
    face_crop = validate_face(cropped_img, face_model, face_meta)
    if not face_crop["is_face"]:
        cropped_img  = pil_img
        crop_success = False
        face_crop    = face_orig

    # Step 3: prediksi skin tone
    prediction = predict_skin_tone(cropped_img, skin_model, skin_meta)
    ambiguity  = check_ambiguity(prediction["probabilities"])

    # Step 4: tentukan tone final
    # Dynamic target selalu dihitung dari probabilitas prediksi AI, baik
    # untuk hasil AI maupun manual correction, agar rekomendasi tetap personal.
    dynamic_target = calculate_dynamic_target(prediction["probabilities"], target_lightness)

    if manual_tone and manual_tone in target_lightness:
        final_tone  = manual_tone
        tone_source = "manual_correction"
    else:
        final_tone  = prediction["predicted_skin_tone"]
        tone_source = "ai_prediction"

    # Step 5: rekomendasikan foundation
    rec = recommend_foundation(
        final_tone, skin_type, df, target_lightness,
        lightness_range, skin_type_rules, top_n,
        target_override=dynamic_target,
    )

    return {
        "status"         : "accepted",
        "original_img"   : pil_img,
        "cropped_img"    : cropped_img,
        "crop_success"   : crop_success,
        "face_orig"      : face_orig,
        "face_crop"      : face_crop,
        "prediction"     : prediction,
        "ambiguity"      : ambiguity,
        "final_skin_tone": final_tone,
        "tone_source"    : tone_source,
        "dynamic_target" : dynamic_target,
        "recommendation" : rec,
    }

def build_final_recommendation(base_result, skin_type_key, manual_tone, top_n,
                                target_lightness, lightness_range, skin_type_rules, df_foundation):
    """
    Rebuild rekomendasi setelah user mengubah manual tone atau jumlah produk.
    Dipanggil tiap kali Customize Results diubah.
    """
    # Dynamic target tetap dari probabilitas AI 
    # skin tone yang berubah saat manual correction dipilih.
    dynamic_target = calculate_dynamic_target(
        base_result["prediction"]["probabilities"], target_lightness
    )

    if manual_tone and manual_tone in target_lightness:
        final_tone  = manual_tone
        tone_source = "manual_correction"
    else:
        final_tone  = base_result["prediction"]["predicted_skin_tone"]
        tone_source = "ai_prediction"

    rec = recommend_foundation(
        final_tone, skin_type_key, df_foundation,
        target_lightness, lightness_range, skin_type_rules,
        top_n, target_override=dynamic_target,
    )
    return {
        **base_result,
        "final_skin_tone": final_tone,
        "tone_source"    : tone_source,
        "dynamic_target" : dynamic_target,
        "recommendation" : rec,
    }

def display_setup_error(missing_files):
    """Tampilkan error jika file model tidak lengkap."""
    st.error("File model belum lengkap di folder `models/`.")
    st.code("\n".join(str(p) for p in missing_files))

def reset_if_new_upload(uploaded_file):
    """Reset hasil analisis jika user upload foto baru."""
    name = uploaded_file.name if uploaded_file else None
    if name != st.session_state.m2s_uploaded_name:
        st.session_state.m2s_result        = None
        st.session_state.m2s_uploaded_img  = None
        st.session_state.m2s_uploaded_name = name

# ============================================================
# INIT — cek file, load model, inisialisasi session state
# ============================================================

# Cek semua file wajib sebelum lanjut
missing = [p for p in REQUIRED_FILES if not p.exists()]
if missing:
    display_setup_error(missing)
    st.stop()

# Load model dan data 
skin_model = load_skin_model()
face_model = load_face_model()
skin_meta, face_meta, target_lightness, lightness_range, skin_type_rules, df_foundation = load_assets()

# Inisialisasi session state untuk menyimpan hasil antar interaksi
for key, default in {
    "m2s_result"       : None,
    "m2s_uploaded_name": None,
    "m2s_uploaded_img" : None,
    "m2s_skin_type"    : None,
    "m2s_top_n"        : DEFAULT_TOP_N,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# TABS
# ============================================================

st.markdown('<div class="tabs-top-spacer"></div>', unsafe_allow_html=True)
tab_match, tab_tips, tab_faq = st.tabs(["Find My Shade", "Makeup Tips", "FAQ"])

# ============================================================
# TAB 1 — FIND MY SHADE
# ============================================================

with tab_match:

    # Header
    if HEADER_IMAGE_PATH.exists():
        header_b64 = base64.b64encode(HEADER_IMAGE_PATH.read_bytes()).decode("utf-8")
        st.markdown(
            f'<div class="header-image-html"><img src="data:image/png;base64,{header_b64}" alt="MatchMyShade header"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="hero-banner main-hero">
            <div class="hero-kicker">AI Foundation Finder</div>
            <div class="hero-title">Match<span>My</span>Shade</div>
            <div class="hero-subtitle">Find your closest foundation shade match based on skin tone and lightness similarity</div>
        </div>""", unsafe_allow_html=True)

    # Panduan cara pakai
    st.markdown("""
    <div class="how-card">
        <div class="section-eyebrow">How it works</div>
        <div class="how-step"><div class="how-step-num">1</div><div class="how-step-text">Upload foto wajah yang jelas dengan pencahayaan natural</div></div>
        <div class="how-step"><div class="how-step-num">2</div><div class="how-step-text">Pilih jenis kulit yang sesuai, seperti oily, dry, combination, normal, atau sensitive.</div></div>
        <div class="how-step"><div class="how-step-num">3</div><div class="how-step-text">AI akan menganalisis skin tone dan memberikan rekomendasi shade foundation yang paling sesuai.</div></div>
    </div>""", unsafe_allow_html=True)

    # Preview 4 kategori skin tone
    st.markdown('<span class="section-eyebrow">Skin tone categories</span>', unsafe_allow_html=True)
    tone_cols = st.columns(4, gap="small")
    for col, (tone, color) in zip(tone_cols, SKIN_TONE_COLORS.items()):
        with col:
            render_tone_box(tone, color)

    st.divider()

    # Form upload foto dan pilih skin type
    st.markdown('<span class="section-eyebrow" style="margin-bottom:10px;">Upload your photo</span>', unsafe_allow_html=True)
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Upload foto wajah",
            type=["jpg", "jpeg", "png", "webp"],
            help="Gunakan foto close-up wajah, tanpa filter ekstrem, dengan cahaya yang cukup.",
            label_visibility="collapsed",
        )
        reset_if_new_upload(uploaded)

        SKIN_TYPE_OPTIONS = [""] + list(SKIN_TYPE_LABELS.keys())

        skin_type_key = st.selectbox(
            "Jenis kulit kamu",
            options=SKIN_TYPE_OPTIONS,
            format_func=lambda k: "— Pilih jenis kulit —" if k == "" else SKIN_TYPE_LABELS[k],
            index=0,
        )
        analyze_btn = st.button(
            "Find My Match",
            type="primary",
            use_container_width=True,
            disabled=(skin_type_key == ""),
        )

    # Placeholder sebelum foto diupload
    if uploaded is None:
        st.markdown(
            '<div style="text-align:center;padding:24px 0 8px;opacity:0.45;font-size:0.9rem;">Upload foto wajah kamu untuk memulai analisis.</div>',
            unsafe_allow_html=True,
        )

    else:
        # Jalankan pipeline saat tombol ditekan
        if analyze_btn:
            if skin_type_key == "":
                st.warning("Pilih jenis kulit dulu ya!")
            else:
                pil_img = Image.open(uploaded).convert("RGB")
                st.session_state.m2s_uploaded_img = pil_img
                st.session_state.m2s_skin_type    = skin_type_key
                with st.spinner("Analyzing your skin tone..."):
                    st.session_state.m2s_result = run_pipeline(
                        pil_img=pil_img, skin_type=skin_type_key,
                        top_n=st.session_state.m2s_top_n,
                        face_model=face_model, face_meta=face_meta,
                        skin_model=skin_model, skin_meta=skin_meta,
                        df=df_foundation, target_lightness=target_lightness,
                        lightness_range=lightness_range, skin_type_rules=skin_type_rules,
                        manual_tone=None,
                    )
        elif st.session_state.m2s_result is None:
            st.info("Klik tombol di atas untuk memulai analisis.")

        # Tampilkan hasil jika sudah ada
        if st.session_state.m2s_result is not None:
            base_result   = st.session_state.m2s_result
            pil_img       = st.session_state.m2s_uploaded_img
            skin_type_key = st.session_state.m2s_skin_type or skin_type_key

            # ── Rejected: wajah tidak terdeteksi ──
            if base_result["status"] == "rejected":
                st.divider()
                st.error(base_result["message"])
                st.markdown(
                    "**Tips agar foto bisa diproses:**\n\n"
                    "- Gunakan foto close-up wajah yang jelas.\n"
                    "- Hindari foto yang terlalu gelap, blur, atau menggunakan filter ekstrem.\n"
                    "- Pastikan wajah menghadap kamera secara langsung.\n"
                    "- Jangan unggah gambar selain wajah manusia, seperti produk, hewan, objek, atau pemandangan."
                )

            # ── Accepted: tampilkan hasil lengkap ──
            else:
                pred      = base_result["prediction"]
                ambiguity = base_result["ambiguity"]
                ai_tone   = pred["predicted_skin_tone"]
                tone_bg   = SKIN_TONE_COLORS.get(ai_tone, "#ddd")
                tone_txt  = "#2C1A1D" if ai_tone in ["Fair", "Medium"] else "#FFF8F8"

                st.divider()

                # Hasil prediksi AI + foto crop
                st.markdown('<span class="section-eyebrow">YOUR SKIN TONE ANALYSIS</span>', unsafe_allow_html=True)
                col_result, col_photo = st.columns([1, 1], gap="large")
                with col_result:
                    st.markdown(f"""
                    <div class="tone-result-wrap" style="background:{tone_bg};color:{tone_txt};">
                        <div class="tone-result-label">YOUR SKIN TONE</div>
                        <div class="tone-result-value">{ai_tone}</div>
                    </div>""", unsafe_allow_html=True)
                with col_photo:
                    st.image(base_result["cropped_img"], caption="Image used for analysis", use_container_width=True)

                # Warning jika prediksi ambigu
                if ambiguity["is_ambiguous"]:
                    t1 = ambiguity["top1"]
                    t2 = ambiguity["top2"]
                    st.warning(
                        f"Hasil AI menunjukkan warna kulit berada di antara "
                        f"**{t1['label']}** dan **{t2['label']}**. "
                        f"Jika hasilnya belum sesuai, kamu bisa menyesuaikannya secara manual di bawah."
                    )

                st.divider()

                # ── Customize Results ──
                st.markdown('<span class="section-eyebrow" style="margin-bottom:8px;">Customize Results</span>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(
                        '<div class="customize-intro">'
                        '<strong>Adjust your recommendation</strong>'
                        'Sesuaikan hasil rekomendasi dan jumlah produk yang ditampilkan.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    # Pilihan: gunakan AI atau override manual
                    st.markdown('<span class="customize-section-label">Apakah prediksi skin tone sudah sesuai?</span>', unsafe_allow_html=True)
                    correction_choice = st.radio(
                        "correction",
                        options=["Gunakan hasil AI", "Pilih sendiri (manual)"],
                        horizontal=False,
                        label_visibility="collapsed",
                    )
                    manual_tone = None
                    if "manual" in correction_choice.lower():
                        other_tones = [t for t in ALL_SKIN_TONES if t != ai_tone]
                        manual_tone = st.selectbox("Pilih skin tone kamu", options=other_tones)

                    st.markdown('<div style="height:10px;border-top:1.5px solid rgba(194,116,138,0.25);margin:10px 0 14px;"></div>', unsafe_allow_html=True)

                    # Pilihan jumlah produk
                    st.markdown('<span class="customize-section-label">Jumlah produk yang ditampilkan</span>', unsafe_allow_html=True)
                    top_n = st.selectbox(
                        "Jumlah rekomendasi",
                        options=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        index=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12].index(int(st.session_state.m2s_top_n))
                        if int(st.session_state.m2s_top_n) in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] else 3,
                        label_visibility="collapsed",
                    )
                    st.caption(f"Menampilkan {int(top_n)} rekomendasi shade foundation teratas.")
                    st.session_state.m2s_top_n = int(top_n)

                final_result = build_final_recommendation(
                    base_result, skin_type_key, manual_tone, int(top_n),
                    target_lightness, lightness_range, skin_type_rules, df_foundation,
                )
                rec        = final_result["recommendation"]
                final_tone = final_result["final_skin_tone"]
                tone_src   = final_result["tone_source"]

                st.divider()

                # ── Match Summary ──
                st.markdown('<span class="section-eyebrow">Match Summary</span>', unsafe_allow_html=True)
                render_match_summary(
                    final_tone=final_tone,
                    skin_type_key=skin_type_key,
                    finish=rec["suggested_finish"],
                    coverage=rec["suggested_coverage"],
                    skin_tone_colors=SKIN_TONE_COLORS,
                    skin_type_labels_short=SKIN_TYPE_LABELS_SHORT,
                )
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                st.caption(f"Source: {'Match My Shade AI' if tone_src == 'ai_prediction' else 'Manual correction by user'}")
                st.markdown(f'<div class="tip-box">{rec["tips"]}</div>', unsafe_allow_html=True)

                st.divider()

                # ── Product Recommendations ──
                st.markdown(f'<span class="section-eyebrow">Top {int(top_n)} Color Matches</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-heading">Your best foundation matches</div>', unsafe_allow_html=True)
                st.caption("Produk diurutkan berdasarkan tingkat kecocokan shade dan lightness. Finish dan coverage yang disarankan disesuaikan dengan jenis kulit.")
                render_product_cards(rec["recommended_products"], final_tone, rec)

                st.divider()

                # ── Shareable Shade Card ──
                st.markdown('<span class="section-eyebrow">Share Your Results</span>', unsafe_allow_html=True)

                preview_products = rec.get("recommended_products", [])[:5]
                dots_html = "".join(
                    f'<div class="share-card-swatch-dot" style="background:{p.get("hex", "#C98F68")};"></div>'
                    for p in preview_products
                )
                st.markdown(
                    '<div class="share-card-panel">'
                    '<div class="share-card-title">Share your shade match ✨</div>'
                    '<div class="share-card-desc">Create a shareable card featuring your Top 5 shade matches. Your uploaded photo is not included.</div>'
                    f'<div class="share-card-swatches">{dots_html}'
                    '<span class="share-card-swatch-label">Your Top 5 palette</span></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                shade_png = create_shade_card_png(final_result, skin_type_key)
                file_tone = str(final_tone).lower().replace(" ", "-")
                st.download_button(
                    "Save My Shade Card",
                    data=shade_png,
                    file_name=f"matchmyshade-{file_tone}-shade-card.png",
                    mime="image/png",
                    use_container_width=True,
                    type="primary",
                )

                st.caption("Save your shade card and share it anywhere.")

                st.divider()

                # Footer model info
                accuracy = skin_meta.get("test_accuracy", 0)
                f1_val   = skin_meta.get("test_f1_score", None)
                footer = (
                    f"Powered by MatchMyShade AI · MobileNetV2 (Fine-tuned) · "
                    f"Test accuracy: {accuracy:.1%}"
                )
                if f1_val:
                    footer += f" · F1-score: {f1_val:.1%}"
                st.caption(footer)

# ============================================================
# TAB 2 — MAKEUP TIPS
# ============================================================

with tab_tips:
    st.markdown("""
    <div class="hero-banner" style="padding:34px 32px;">
        <div class="hero-title" style="font-size:2.35rem;">Makeup <span>Tips</span></div>
        <div class="hero-subtitle">Panduan memilih dan memakai foundation dengan lebih tepat</div>
    </div>
    """, unsafe_allow_html=True)

    # Card 1: Cara memilih shade 
    st.markdown("""
    <div class="tips-section-wrap">
      <div class="tips-card">
        <div class="tips-card-header">
          <div class="tips-card-icon">✦</div>
          <h3>Cara Memilih Shade yang Tepat</h3>
        </div>
        <div class="tips-list">
          <div class="tips-row"><div class="tips-dot"></div><div><strong>Jaw test:</strong> Coba shade di area rahang agar warna lebih mudah dibandingkan dengan wajah dan leher.</div></div>
          <div class="tips-row"><div class="tips-dot"></div><div><strong>Natural light:</strong> Cek hasil foundation di cahaya alami supaya warna tidak terlihat terlalu terang, abu-abu, atau oranye.</div></div>
          <div class="tips-row"><div class="tips-dot"></div><div><strong>Oksidasi:</strong> Tunggu 5 sampai 10 menit setelah pemakaian, karena beberapa foundation bisa berubah sedikit lebih gelap di kulit.</div></div>
          <div class="tips-row"><div class="tips-dot"></div><div><strong>Shade lighter:</strong> Jika foundation sering menggelap setelah dipakai, pilih shade setengah tingkat lebih terang.</div></div>
          <div class="tips-row"><div class="tips-dot"></div><div><strong>Neck match:</strong> Jika ragu antara dua shade, pilih warna yang paling mendekati leher agar hasilnya terlihat natural.</div></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: Tips per jenis kulit 
    skin_order = ["oily", "dry", "combination", "sensitive", "normal"]

    skin_items_html = ""
    for i, key in enumerate(skin_order):
        rule  = skin_type_rules.get(key, {})
        label = SKIN_TYPE_LABELS_SHORT.get(key, key.capitalize())
        tips  = normalize_tips(rule.get("tips", ""))
        finish   = rule.get("suggested_finish", rule.get("finish", ""))
        coverage = rule.get("suggested_coverage", rule.get("coverage", ""))

        span = 'style="grid-column: span 2;"' if key == "normal" else ""
        skin_items_html += f"""
        <div class="tips-skin-item" {span}>
            <div class="tips-skin-label">{label}</div>
            <div class="tips-skin-desc">{tips}</div>
            <div class="tips-skin-meta">
                <span class="prod-pill prod-pill-finish">{finish}</span>
                <span class="prod-pill prod-pill-coverage">{coverage}</span>
            </div>
        </div>"""

    st.markdown(f"""
    <div class="tips-section-wrap">
      <div class="tips-card">
        <div class="tips-card-header">
          <div class="tips-card-icon">✦</div>
          <h3>Tips per Jenis Kulit</h3>
        </div>
        <div class="tips-skin-grid">{skin_items_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Disclaimer
    st.markdown("""
    <div class="tips-section-wrap">
      <div class="tips-card tips-card-disclaimer">
        <div class="tips-card-header">
          <div class="tips-card-icon">✦</div>
          <h3>Disclaimer</h3>
        </div>
        <p style="font-size:0.95rem;line-height:1.75;margin:0;opacity:0.85;">
          Rekomendasi produk ditentukan berdasarkan kecocokan warna/lightness, bukan berdasarkan formula produk.
          Saran finish dan coverage merupakan panduan tambahan sesuai jenis kulit.
          Hasil rekomendasi dapat dipengaruhi oleh pencahayaan foto, kualitas kamera, penggunaan filter, dan ketersediaan data produk.
          Gunakan rekomendasi ini sebagai referensi awal, lalu cek shade secara langsung jika memungkinkan.
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 3 — FAQ
# ============================================================

FAQ_ITEMS = [
    ("Apakah foto saya aman dan tidak disimpan?",
     "Iya, aman. Foto kamu hanya diproses sementara di memori selama sesi berlangsung, setelah sesi selesai atau browser ditutup, foto langsung hilang. Kami tidak menyimpan foto, identitas, atau data apapun ke server maupun database. Tidak ada yang melihat foto kamu selain kamu sendiri."),

    ("Apakah hasil deteksi skin tone selalu akurat?",
     "Tidak selalu, dan itu wajar. Hasil bisa dipengaruhi oleh pencahayaan ruangan, kualitas kamera, penggunaan filter, atau sudut wajah saat foto diambil. Untuk hasil terbaik, gunakan foto wajah yang jelas dengan cahaya natural tanpa filter. Jika hasil AI dirasa kurang tepat, kamu bisa menggunakan opsi koreksi manual untuk memilih skin tone sendiri."),

    ("Apa saja kategori skin tone yang digunakan?",
     "MatchMyShade menggunakan 4 kategori skin tone: Fair (cerah), Medium (sedang), Tan (sawo matang), dan Deep (gelap). Kategori ini ditentukan berdasarkan nilai lightness warna kulit dari foto yang diunggah. Setiap kategori memiliki kisaran lightness shade foundation yang berbeda agar rekomendasinya lebih relevan."),

    ("Kenapa rekomendasi AI dan koreksi manual bisa berbeda?",
     "Rekomendasi AI dibuat khusus berdasarkan foto yang kamu upload, jadi hasilnya unik untuk kamu. Kalau kamu pakai koreksi manual, kamu memilih sendiri kategori skin tone-nya — tapi rekomendasinya tetap disesuaikan dengan warna kulitmu dari foto, bukan hasil yang generik."),

    ("Kenapa link produk ada yang tidak bisa dibuka?",
     "Beberapa link di tombol View Product berasal dari dataset dan mungkin sudah tidak aktif atau berubah. Karena itu kami juga menyediakan tombol Search Online yang langsung mencari produk tersebut di Google sebagai alternatif."),

    ("Haruskah saya tetap coba shade langsung?",
     "Sangat disarankan. MatchMyShade membantu mempersempit pilihan shade berdasarkan kecocokan warna, tapi hasil akhirnya tetap bisa berbeda di kulit masing-masing karena faktor undertone, oksidasi formula, dan kondisi kulit. Gunakan rekomendasi ini sebagai titik awal, lalu cek langsung jika memungkinkan."),

    ("Kenapa wajah saya tidak terdeteksi?",
     "Pastikan foto menampilkan wajah secara jelas, menghadap kamera, dengan pencahayaan yang cukup. Hindari foto yang terlalu gelap, blur, atau menggunakan filter ekstrem. Foto selfie close-up tanpa filter biasanya memberikan hasil terbaik."),
]

with tab_faq:
    st.markdown("""
    <div class="hero-banner" style="padding:34px 32px;">
        <div class="hero-title" style="font-size:2.35rem;">Frequently <span>Asked</span></div>
        <div class="hero-subtitle">Pertanyaan umum tentang MatchMyShade</div>
    </div>
    """, unsafe_allow_html=True)
    render_faq(FAQ_ITEMS)