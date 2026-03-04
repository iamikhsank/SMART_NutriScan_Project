import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
import plotly.express as px
import plotly.graph_objects as go

import scipy.linalg

# Patch scipy.linalg.triu for gensim compatibility
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu

# Import the new model utility functions
from model_utils import load_prediction_models, analyze_product_fully, preprocess_batch_excel_data

import easyocr

# Initialize session state for history
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SMART NutriScan AI",
    page_icon="assets/Logo Smart NutriScan AI.png",
    layout="wide"
)

# --- Memuat Model ---
# Use the new loading function from model_utils.py
@st.cache_resource
def load_all_models_and_scaler():
    """Fungsi untuk memuat semua model AI dan scaler."""
    feat_model, lgbm_model, w2v_model, scaler = load_prediction_models()
    if feat_model and lgbm_model and w2v_model and scaler:
        st.success("Model AI dan scaler berhasil dimuat.")
        return feat_model, lgbm_model, w2v_model, scaler
    else:
        st.error("Gagal memuat satu atau lebih komponen AI. Aplikasi mungkin tidak berfungsi dengan benar.")
        return None, None, None, None

@st.cache_resource
def load_ocr_model():
    """Fungsi untuk memuat model OCR."""
    reader = easyocr.Reader(['id', 'en']) # 'id' for Indonesian, 'en' for English
    st.success("Model OCR (EasyOCR) berhasil dimuat.")
    return reader

feat_model, lgbm_model, w2v_model, scaler = load_all_models_and_scaler()
reader = load_ocr_model()



def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Analisis')
    processed_data = output.getvalue()
    return processed_data


def parse_nutrition_text(text):


    """


    Menganalisis teks OCR untuk mengekstrak informasi nutrisi dan komposisi.


    Menggunakan regex yang lebih fleksibel.


    """


    # Normalisasi teks: ganti koma desimal, hapus spasi berlebih


    text = text.replace(',', '.').lower()


    text = re.sub(r'\s+', ' ', text)





    # Pola regex yang lebih kuat


    patterns = {


        'energi': r"(?:energi|energy)\s*(?:dari\s*lemak)?\s*:?\s*(\d+(?:\.\d+)?)",


        'lemak_total': r"(?:lemak|fat)\s*total\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'lemak_jenuh': r"(?:lemak|fat)\s*jenuh\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'protein': r"protein\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'karbohidrat': r"karbohidrat\s*total\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'gula': r"gula\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'garam': r"garam\s*(?:\(natrium\))?\s*:?\s*(\d+(?:\.\d+)?)\s*g",


        'natrium': r"natrium\s*(?:/sodium)?\s*:?\s*(\d+)\s*mg",

        'natrium_benzoat': r"natrium\s*benzoat\s*:?\s*(\d+(?:\.\d+)?)\s*mg"


    }


    


    data = {}


    # Ekstraksi nilai nutrisi


    for key, pattern in patterns.items():


        match = re.search(pattern, text)


        if match:


            try:


                data[key] = float(match.group(1))


            except (ValueError, IndexError):


                data[key] = 0.0


        else:


            data[key] = 0.0





    # Logika fallback untuk Natrium dari Garam


    if data.get('garam', 0) > 0 and data.get('natrium', 0) == 0:


        data['natrium'] = data['garam'] * 400  # 1g garam ~= 400mg natrium





    # Ekstraksi komposisi yang lebih canggih


    komposisi_match = re.search(r"(?:komposisi|ingredients|daftar bahan)\s*:\s*(.*?)(?:\.|$)", text)


    if komposisi_match:


        # Mengambil semua teks setelah 'komposisi:' sampai titik atau akhir baris


        komposisi_text = komposisi_match.group(1).strip()


        # Membersihkan dari info alergen yang mungkin menempel


        komposisi_text = re.split(r"mengandung alergen", komposisi_text, flags=re.IGNORECASE)[0]


        data['komposisi'] = komposisi_text.strip().capitalize()


    else:


        data['komposisi'] = "Tidak terdeteksi."


        


    # Ekstraksi nama produk (jika memungkinkan)


    # Ini adalah heuristik sederhana dan mungkin perlu disesuaikan


    product_name_match = re.search(r"^(.*?)\s*(?:informasi nilai gizi|nutrition facts)", text)


    if product_name_match:


        data['product_name'] = product_name_match.group(1).strip().title()


    else:


        # Fallback jika pola utama tidak ditemukan


        lines = text.split('\n')


        if lines:


            data['product_name'] = lines[0].strip().title()


        else:


            data['product_name'] = "Produk Tanpa Nama"





    return data

# --- FUNGSI HELPER UNTUK VISUALISASI BI & HEALTH METRICS ---
def render_holistic_nutrition_metrics(energi, takaran_saji, lemak_total, karbohidrat, protein, gula, natrium, lemak_jenuh, current_threshold, user_profile):
    st.markdown("---")
    st.markdown("### 📊 Profil Gizi & Makronutrien Holistik")
    st.write("Analisis mendalam mengenai sumber kalori dan dampak glikemik berdasarkan takaran saji.")

    # 1. Row Atas: Kepadatan Energi & Rasio Glikemik
    metrik_col1, metrik_col2 = st.columns(2)

    with metrik_col1:
        kepadatan_energi = energi / takaran_saji if takaran_saji > 0 else 0

        # Klasifikasi Kepadatan Energi
        if kepadatan_energi > 4:
            kepadatan_status = "🔴 Sangat Tinggi (Padat Kalori)"
            kepadatan_color = "inverse"
        elif kepadatan_energi > 1.5:
            kepadatan_status = "🟡 Tinggi"
            kepadatan_color = "off"
        elif kepadatan_energi > 0.6:
            kepadatan_status = "🟢 Rendah (Ideal)"
            kepadatan_color = "normal"
        else:
            kepadatan_status = "🔵 Sangat Rendah"
            kepadatan_color = "normal"

        st.metric(label="Kepadatan Energi (kkal/gram)", value=f"{kepadatan_energi:.1f}", delta=kepadatan_status, delta_color=kepadatan_color)
        st.caption("Menunjukkan seberapa padat kalori dalam produk ini. Kepadatan tinggi memicu obesitas jika tidak dikontrol.")

    with metrik_col2:
        rasio_gula = (gula / karbohidrat) * 100 if karbohidrat > 0 else 0
        if rasio_gula > 50:
            rasio_status = "🔴 Tinggi Gula Sederhana"
            rasio_color = "inverse"
        elif rasio_gula > 25:
            rasio_status = "🟡 Waspada Glikemik"
            rasio_color = "off"
        else:
            rasio_status = "🟢 Karbohidrat Kompleks"
            rasio_color = "normal"

        st.metric(label="Rasio Gula dari Total Karbohidrat", value=f"{rasio_gula:.1f}%", delta=rasio_status, delta_color=rasio_color)
        st.caption("Jika >50%, sebagian besar karbohidrat adalah gula sederhana yang bisa memicu lonjakan gula darah (*sugar spike*).")

    # 2. Row Tengah: Distribusi Makronutrien (Pie Chart)
    kalori_lemak = lemak_total * 9
    kalori_karbo = karbohidrat * 4
    kalori_protein = protein * 4
    total_kalori_makro = kalori_lemak + kalori_karbo + kalori_protein

    if total_kalori_makro > 0:
        df_makro = pd.DataFrame({
            "Sumber": ["Lemak (9 kkal/g)", "Karbohidrat (4 kkal/g)", "Protein (4 kkal/g)"],
            "Kalori": [kalori_lemak, kalori_karbo, kalori_protein],
            "Gram": [lemak_total, karbohidrat, protein]
        })

        fig_makro = px.pie(
            df_makro,
            values="Kalori",
            names="Sumber",
            hole=0.45,
            color="Sumber",
            color_discrete_map={
                "Lemak (9 kkal/g)": "#EF553B",
                "Karbohidrat (4 kkal/g)": "#00CC96",
                "Protein (4 kkal/g)": "#636EFA"
            },
            title="Distribusi Sumber Kalori (Macronutrient Split)",
            hover_data=['Gram']
        )
        fig_makro.update_traces(textposition='inside', textinfo='percent+label')
        fig_makro.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_makro, use_container_width=True)

    # 3. Row Bawah: Progress Bar AKG (Angka Kecukupan Gizi) per Takaran Saji
    st.markdown("#### Pemenuhan Angka Kecukupan Gizi (AKG) Harian")
    st.write(f"Persentase batas harian profil **{user_profile}** yang terpakai untuk **1 Takaran Saji ({takaran_saji}g/ml)** produk ini:")

    # Gula
    pct_gula = (gula / current_threshold['gula']) * 100 if current_threshold['gula'] > 0 else 0
    st.write(f"**Gula**: {gula}g dari batas {current_threshold['gula']}g/hari")
    st.progress(min(int(pct_gula), 100))
    if pct_gula > 50:
        st.warning(f"⚠️ 1 Porsi produk ini menghabiskan **{pct_gula:.1f}%** jatah gula harian Anda!")

    # Natrium
    pct_natrium = (natrium / current_threshold['natrium']) * 100 if current_threshold['natrium'] > 0 else 0
    st.write(f"**Natrium**: {natrium}mg dari batas {current_threshold['natrium']}mg/hari")
    st.progress(min(int(pct_natrium), 100))
    if pct_natrium > 50:
        st.warning(f"⚠️ 1 Porsi produk ini menghabiskan **{pct_natrium:.1f}%** jatah natrium harian Anda!")

    # Lemak Jenuh
    pct_lemak_jenuh = (lemak_jenuh / current_threshold['lemak_jenuh']) * 100 if current_threshold['lemak_jenuh'] > 0 else 0
    st.write(f"**Lemak Jenuh**: {lemak_jenuh}g dari batas {current_threshold['lemak_jenuh']}g/hari")
    st.progress(min(int(pct_lemak_jenuh), 100))
    if pct_lemak_jenuh > 50:
        st.warning(f"⚠️ 1 Porsi produk ini menghabiskan **{pct_lemak_jenuh:.1f}%** jatah lemak jenuh harian Anda!")


# --- UI Aplikasi ---

# --- Sidebar ---
with st.sidebar:
    st.image("assets/Logo Smart NutriScan AI.png", width=150)
    st.title("SMART NutriScan AI")
    
    st.header("5. Profil Pengguna")
    user_profile = st.selectbox(
        "Pilih profil kesehatan Anda:",
        ("Dewasa", "Anak-anak", "Lansia", "Penderita Hipertensi", "Risiko Penyakit Ginjal")
    )
    
    # Batas harian sekarang mencakup kebutuhan total sebagai referensi BI metrics
    # Nilai Gula (g), Natrium (mg), Lemak Jenuh (g) per hari
    daily_limits = {
        "Dewasa": {"gula": 50, "natrium": 2000, "lemak_jenuh": 22},
        "Anak-anak": {"gula": 25, "natrium": 1500, "lemak_jenuh": 16},
        "Lansia": {"gula": 30, "natrium": 1500, "lemak_jenuh": 20},
        "Penderita Hipertensi": {"gula": 25, "natrium": 1200, "lemak_jenuh": 18},
        "Risiko Penyakit Ginjal": {"gula": 25, "natrium": 1000, "lemak_jenuh": 18},
    }
    current_threshold = daily_limits[user_profile]

    st.markdown("---")
    
    app_mode = st.radio(
        "Pilih Fitur:",
        ["Analisis Produk Tunggal", "Scan from Image", "Analisis Batch (Excel)", "Perbandingan Produk", "Simulasi Konsumsi", "Riwayat Analisis", "Edukasi Gizi"]
    )
    st.markdown("---")
    st.info("Dashboard ini adalah sistem intelijen terpadu. Fitur Analisis AI di-back-up oleh model hybrid CBLIGHT-WOA & BI Analytics.")

# --- Halaman Utama ---

if app_mode == "Analisis Produk Tunggal":
    st.header("Analisis Produk Pangan dengan AI")

    if not all([feat_model, lgbm_model, w2v_model, scaler]):
        st.error("Model tidak dapat digunakan. Silakan periksa log kesalahan di konsol.")
    else:
        st.success("Model AI aktif dan siap digunakan untuk analisis.")
        st.markdown("---")

        main_col, right_col = st.columns([1.8, 1.2])

        with main_col:
            st.subheader("Input Informasi Produk")
            st.markdown("Isi form di bawah ini dengan informasi dari label nutrisi produk.")
            
            product_name = st.text_input("Nama Produk", "Biskuit Cokelat")

            # Form untuk input data (Ditambah Takaran Saji untuk BI)
            c0, c1, c2 = st.columns(3)
            takaran_saji = c0.number_input("Takaran Saji (g/ml)", min_value=1.0, value=30.0, format="%.1f")
            energi = c1.number_input("Energi (kkal)", min_value=0, value=180)
            lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=8.0, format="%.1f")

            c3, c4, c5 = st.columns(3)
            lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=4.0, format="%.1f")
            protein = c4.number_input("Protein (g)", min_value=0.0, value=2.0, format="%.1f")
            karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=25.0, format="%.1f")

            c6, c7, c8, c9 = st.columns(4)
            gula = c6.number_input("Gula (g)", min_value=0.0, value=15.0, format="%.1f")
            garam = c7.number_input("Garam (g)", min_value=0.0, value=0.3, format="%.2f")
            natrium = c8.number_input("Natrium (mg)", min_value=0, value=200)
            natrium_benzoat = c9.number_input("Natrium Benzoat (mg)", min_value=0.0, value=0.0, format="%.2f")

            komposisi = st.text_area("Komposisi / Ingredients", "Tepung Terigu, Gula, Minyak Nabati, Cokelat Bubuk, Pengembang, Perisa Sintetik, Garam.")

            analyze_button = st.button("✨ Analisis AI & Gizi Sekarang!", type="primary")

        with right_col:
            st.subheader("Hasil Analisis AI (Prediksi Risiko)")
            if analyze_button:
                with st.spinner('Menganalisis produk dengan model CBLIGHT-WOA...'):
                    # Data dictionary ini TETAP murni untuk model ML, tanpa diganggu metrik BI
                    nutrition_data = {
                        'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                        'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                        'garam': garam, 'natrium': natrium, 'natrium_benzoat': natrium_benzoat
                    }

                    risk_score, xai_factors, recommendation = analyze_product_fully(
                        nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
                    )

                    from datetime import datetime
                    st.session_state.scan_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "product_name": product_name,
                        "risk_score": risk_score,
                        "profile": user_profile,
                        "nutrition": nutrition_data
                    })

                    st.metric(label="Skor Risiko Prediksi ML", value=f"{risk_score:.2f}%")
                    if risk_score > 75:
                        st.error("🔴 Risiko Sangat Tinggi")
                    elif risk_score > 50:
                        st.warning("🟠 Risiko Tinggi")
                    elif risk_score > 25:
                        st.warning("🟡 Risiko Sedang")
                    else:
                        st.success("🟢 Risiko Rendah")

                    st.markdown("---")
                    st.markdown("#### Radar Kontribusi Nutrisi (XAI)")

                    categories = list(xai_factors.keys())
                    values = list(xai_factors.values())

                    norm_values = []
                    for k, v in xai_factors.items():
                        if 'gula' in k.lower(): norm_values.append(min((v / 50) * 100, 100))
                        elif 'natrium' in k.lower() and 'benzoat' not in k.lower(): norm_values.append(min((v / 1500) * 100, 100))
                        elif 'lemak' in k.lower(): norm_values.append(min((v / 67) * 100, 100))
                        elif 'energi' in k.lower(): norm_values.append(min((v / 2000) * 100, 100))
                        else: norm_values.append(min((v / 100) * 100, 100))

                    fig_radar = go.Figure()

                    fig_radar.add_trace(go.Scatterpolar(
                        r=norm_values + [norm_values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='Kandungan Produk',
                        line_color='red' if risk_score > 50 else 'orange' if risk_score > 25 else 'green',
                        hovertemplate="Feature: %{theta}<br>Skor Relatif: %{r:.1f}/100<extra></extra>"
                    ))

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
                        showlegend=False,
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=250
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    st.markdown("#### Rekomendasi ML")
                    st.info(recommendation)

            else:
                st.metric(label="Skor Risiko Prediksi", value="-")
                st.info("Input data dan jalankan analisis untuk melihat detail prediksi risiko, Radar XAI, dan Rekomendasi ML.")

        # Eksekusi visualisasi Business & Health Metrics di bawah setelah tombol di-klik
        if analyze_button:
            render_holistic_nutrition_metrics(energi, takaran_saji, lemak_total, karbohidrat, protein, gula, natrium, lemak_jenuh, current_threshold, user_profile)


elif app_mode == "Scan from Image":
    st.header("1. Scan Produk Otomatis melalui Foto")
    st.info("Unggah gambar kemasan produk. Sistem OCR akan membaca informasi gizi, dilanjutkan dengan analitik AI & BI.")

    uploaded_image = st.file_uploader("Pilih gambar produk...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        col_img, col_proc = st.columns([1, 2])
        with col_img:
            st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

        with col_proc:
            with st.spinner("Membaca dan menganalisis teks dari gambar..."):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                ocr_results = reader.readtext(img_byte_arr, detail=0, paragraph=True)
                detected_text = " ".join(ocr_results)
                parsed_data = parse_nutrition_text(detected_text)

            st.success("✨ Teks berhasil dibaca! Silakan lengkapi 'Takaran Saji' dan koreksi data jika perlu.")

        st.markdown("---")

        main_col, right_col = st.columns([1.8, 1.2])

        with main_col:
            st.subheader("Input Informasi Produk (Hasil OCR)")
            
            product_name = st.text_input("Nama Produk", value=parsed_data.get('product_name', ''))

            # Tambahan input takaran saji karena OCR sering miss bagian ini jika formatnya aneh
            c0, c1, c2 = st.columns(3)
            takaran_saji = c0.number_input("Takaran Saji (g/ml)", min_value=1.0, value=30.0, format="%.1f", help="OCR sulit menangkap ini. Mohon isi manual.")
            energi = c1.number_input("Energi (kkal)", min_value=0, value=int(parsed_data.get('energi', 0)))
            lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=parsed_data.get('lemak_total', 0.0), format="%.1f")

            c3, c4, c5 = st.columns(3)
            lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=parsed_data.get('lemak_jenuh', 0.0), format="%.1f")
            protein = c4.number_input("Protein (g)", min_value=0.0, value=parsed_data.get('protein', 0.0), format="%.1f")
            karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=parsed_data.get('karbohidrat', 0.0), format="%.1f")

            c6, c7, c8, c9 = st.columns(4)
            gula = c6.number_input("Gula (g)", min_value=0.0, value=parsed_data.get('gula', 0.0), format="%.1f")
            garam = c7.number_input("Garam (g)", min_value=0.0, value=parsed_data.get('garam', 0.0), format="%.2f")
            natrium = c8.number_input("Natrium (mg)", min_value=0, value=int(parsed_data.get('natrium', 0)))
            natrium_benzoat = c9.number_input("Natrium Benzoat (mg)", min_value=0.0, value=parsed_data.get('natrium_benzoat', 0.0), format="%.2f")

            komposisi = st.text_area("Komposisi / Ingredients", value=parsed_data.get('komposisi', ''), height=100)

            analyze_button = st.button("✨ Analisis AI & Gizi Sekarang!", type="primary")

        with right_col:
            st.subheader("Hasil Analisis AI (Prediksi Risiko)")
            if analyze_button:
                with st.spinner('Menganalisis produk dengan model CBLIGHT-WOA...'):
                    nutrition_data = {
                        'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                        'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                        'garam': garam, 'natrium': natrium, 'natrium_benzoat': natrium_benzoat
                    }

                    risk_score, xai_factors, recommendation = analyze_product_fully(
                        nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
                    )

                    from datetime import datetime
                    st.session_state.scan_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "product_name": product_name,
                        "risk_score": risk_score,
                        "profile": user_profile,
                        "nutrition": nutrition_data
                    })

                    st.metric(label="Skor Risiko Prediksi", value=f"{risk_score:.2f}%")
                    if risk_score > 75: st.error("🔴 Risiko Sangat Tinggi")
                    elif risk_score > 50: st.warning("🟠 Risiko Tinggi")
                    elif risk_score > 25: st.warning("🟡 Risiko Sedang")
                    else: st.success("🟢 Risiko Rendah")

                    st.markdown("---")
                    st.markdown("#### Radar Kontribusi Nutrisi (XAI)")

                    categories = list(xai_factors.keys())
                    values = list(xai_factors.values())

                    norm_values = []
                    for k, v in xai_factors.items():
                        if 'gula' in k.lower(): norm_values.append(min((v / 50) * 100, 100))
                        elif 'natrium' in k.lower() and 'benzoat' not in k.lower(): norm_values.append(min((v / 1500) * 100, 100))
                        elif 'lemak' in k.lower(): norm_values.append(min((v / 67) * 100, 100))
                        elif 'energi' in k.lower(): norm_values.append(min((v / 2000) * 100, 100))
                        else: norm_values.append(min((v / 100) * 100, 100))

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=norm_values + [norm_values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='Kandungan Produk',
                        line_color='red' if risk_score > 50 else 'orange' if risk_score > 25 else 'green',
                        hovertemplate="Feature: %{theta}<br>Skor Relatif: %{r:.1f}/100<extra></extra>"
                    ))

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
                        showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=250
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    st.markdown("#### Rekomendasi ML")
                    st.info(recommendation)
            else:
                 st.metric(label="Skor Risiko Prediksi", value="-")
                 st.info("Jalankan analisis untuk melihat hasil AI.")

        # Panggil render module BI
        if analyze_button:
            render_holistic_nutrition_metrics(energi, takaran_saji, lemak_total, karbohidrat, protein, gula, natrium, lemak_jenuh, current_threshold, user_profile)








elif app_mode == "Analisis Batch (Excel)":
    st.header("Analisis Batch Produk dari File Excel")
    st.write("Unggah file Excel dengan daftar produk dan informasi nutrisinya untuk dianalisis secara bersamaan.")
    
    expected_columns = [
        'Energi', 'Lemak', 'Karbohidrat', 'Gula', 'Protein', 'Garam', 'Komposisi'
    ]
    
    st.info(f"Pastikan file Excel Anda memiliki kolom: {', '.join(expected_columns)}")

    uploaded_excel = st.file_uploader("Pilih file .xlsx", type=["xlsx"])
    
    if uploaded_excel:
        df = pd.read_excel(uploaded_excel)
        
        # Preprocess the batch data to clean units and handle decimal separators
        df = preprocess_batch_excel_data(df)
        
        st.dataframe(df)
        
        # Verify columns exist
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"File Excel tidak memiliki kolom yang dibutuhkan: {', '.join(missing_cols)}")
        else:
            batch_analyze_button = st.button("Mulai Analisis Batch", type="primary", disabled=not all([feat_model, lgbm_model, w2v_model, scaler]))
            
            if batch_analyze_button:
                results = []
                total_rows = len(df)
                progress_bar = st.progress(0)
                
                with st.spinner(f"Menganalisis {total_rows} produk..."):
                    for i, row in df.iterrows():
                        nutrition_data = {
                            'energi': float(row.get('Energi', 0)),
                            'lemak_total': float(row.get('Lemak', 0)),
                            'karbohidrat': float(row.get('Karbohidrat', 0)),
                            'gula': float(row.get('Gula', 0)),
                            'protein': float(row.get('Protein', 0)),
                            'garam': float(row.get('Garam', 0)),
                            'natrium_benzoat': float(row.get('Natrium Benzoat', 0))
                        }
                        composition_text = row.get('Komposisi', "")

                        risk_score, _, _ = analyze_product_fully(
                            nutrition_data, composition_text, feat_model, lgbm_model, w2v_model, scaler
                        )
                        results.append(risk_score)
                        progress_bar.progress((i + 1) / total_rows)
                
                st.success(f"Analisis batch selesai untuk {total_rows} produk!")
                
                df_results = df.copy()
                df_results['Risk Score (%)'] = [f"{r:.2f}" for r in results]
                
                st.subheader("Hasil Analisis Batch")
                st.dataframe(df_results)
                
                df_xlsx = to_excel(df_results)
                st.download_button(
                    label="📥 Download Hasil Analisis (.xlsx)",
                    data=df_xlsx,
                    file_name="hasil_analisis_batch.xlsx"
                )

elif app_mode == "Perbandingan Produk":
    st.header("7. Perbandingan Produk (Food Comparison Mode)")
    st.info("Bandingkan metrik AI (Skor Risiko) dan metrik BI (Kepadatan Energi) dari dua produk sekaligus.")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Produk A")
        p_a_name = st.text_input("Nama Produk A", "Sereal Pagi A")
        p_a_takaran = st.number_input("Takaran Saji A (g)", min_value=1.0, value=30.0, format="%.1f")
        p_a_energi = st.number_input("Energi A (kkal)", min_value=0, value=150)
        p_a_lemak_total = st.number_input("Lemak Total A (g)", min_value=0.0, value=5.0, format="%.1f")
        p_a_lemak_jenuh = st.number_input("Lemak Jenuh A (g)", min_value=0.0, value=1.0, format="%.1f")
        p_a_protein = st.number_input("Protein A (g)", min_value=0.0, value=3.0, format="%.1f")
        p_a_karbohidrat = st.number_input("Karbohidrat A (g)", min_value=0.0, value=30.0, format="%.1f")
        p_a_gula = st.number_input("Gula A (g)", min_value=0.0, value=12.0, format="%.1f")
        p_a_natrium = st.number_input("Natrium A (mg)", min_value=0, value=180)
        p_a_natrium_benzoat = st.number_input("Natrium Benzoat A (mg)", min_value=0.0, value=0.0, format="%.2f")
        p_a_komposisi = st.text_area("Komposisi A", "Gandum Utuh, Gula, Garam.", height=100)
        p_a_garam = p_a_natrium / 400 

    with col2:
        st.subheader("Produk B")
        p_b_name = st.text_input("Nama Produk B", "Sereal Pagi B")
        p_b_takaran = st.number_input("Takaran Saji B (g)", min_value=1.0, value=30.0, format="%.1f")
        p_b_energi = st.number_input("Energi B (kkal)", min_value=0, value=160)
        p_b_lemak_total = st.number_input("Lemak Total B (g)", min_value=0.0, value=6.0, format="%.1f")
        p_b_lemak_jenuh = st.number_input("Lemak Jenuh B (g)", min_value=0.0, value=3.0, format="%.1f")
        p_b_protein = st.number_input("Protein B (g)", min_value=0.0, value=2.0, format="%.1f")
        p_b_karbohidrat = st.number_input("Karbohidrat B (g)", min_value=0.0, value=28.0, format="%.1f")
        p_b_gula = st.number_input("Gula B (g)", min_value=0.0, value=18.0, format="%.1f")
        p_b_natrium = st.number_input("Natrium B (mg)", min_value=0, value=250)
        p_b_natrium_benzoat = st.number_input("Natrium Benzoat B (mg)", min_value=0.0, value=0.0, format="%.2f")
        p_b_komposisi = st.text_area("Komposisi B", "Jagung, Gula, Sirup Fruktosa, Garam.", height=100)
        p_b_garam = p_b_natrium / 400

    st.markdown("---")
    compare_button = st.button("⚖️ Bandingkan Sekarang!", type="primary")

    if compare_button:
        with st.spinner("Menganalisis dan membandingkan kedua produk..."):
            nutrition_a = {
                'energi': p_a_energi, 'lemak_total': p_a_lemak_total, 'lemak_jenuh': p_a_lemak_jenuh,
                'protein': p_a_protein, 'karbohidrat': p_a_karbohidrat, 'gula': p_a_gula,
                'garam': p_a_garam, 'natrium': p_a_natrium, 'natrium_benzoat': p_a_natrium_benzoat
            }
            nutrition_b = {
                'energi': p_b_energi, 'lemak_total': p_b_lemak_total, 'lemak_jenuh': p_b_lemak_jenuh,
                'protein': p_b_protein, 'karbohidrat': p_b_karbohidrat, 'gula': p_b_gula,
                'garam': p_b_garam, 'natrium': p_b_natrium, 'natrium_benzoat': p_b_natrium_benzoat
            }

            risk_a, _, _ = analyze_product_fully(nutrition_a, p_a_komposisi, feat_model, lgbm_model, w2v_model, scaler)
            risk_b, _, _ = analyze_product_fully(nutrition_b, p_b_komposisi, feat_model, lgbm_model, w2v_model, scaler)

            st.subheader("Pemenang Analisis Keseluruhan")
            
            # Kalkulasi Kepadatan Energi untuk BI Metric
            kepadatan_a = p_a_energi / p_a_takaran
            kepadatan_b = p_b_energi / p_b_takaran

            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"### {p_a_name}")
                st.metric(label="Skor Risiko AI", value=f"{risk_a:.2f}%")
                st.metric(label="Kepadatan Energi", value=f"{kepadatan_a:.2f} kkal/g")

            with res_col2:
                st.markdown(f"### {p_b_name}")
                st.metric(label="Skor Risiko AI", value=f"{risk_b:.2f}%")
                st.metric(label="Kepadatan Energi", value=f"{kepadatan_b:.2f} kkal/g")

            st.markdown("---")

            if risk_a < risk_b:
                st.success(f"🏆 **{p_a_name}** adalah pilihan yang lebih baik secara algoritma AI dengan skor risiko lebih rendah.")
            elif risk_b < risk_a:
                st.success(f"🏆 **{p_b_name}** adalah pilihan yang lebih baik secara algoritma AI dengan skor risiko lebih rendah.")
            else:
                st.info("Kedua produk memiliki skor risiko yang sama.")

            st.subheader("Visualisasi Perbandingan Nutrisi (Normalisasi per 100g)")
            # Mengubah ke format per 100g agar perbandingannya apple-to-apple dalam grafik
            faktor_a = 100 / p_a_takaran
            faktor_b = 100 / p_b_takaran

            df_compare = pd.DataFrame({
                "Nutrisi (per 100g)": ["Gula (g)", "Natrium (mg)", "Lemak Jenuh (g)", "Karbohidrat (g)", "Protein (g)"],
                p_a_name: [p_a_gula * faktor_a, p_a_natrium * faktor_a, p_a_lemak_jenuh * faktor_a, p_a_karbohidrat * faktor_a, p_a_protein * faktor_a],
                p_b_name: [p_b_gula * faktor_b, p_b_natrium * faktor_b, p_b_lemak_jenuh * faktor_b, p_b_karbohidrat * faktor_b, p_b_protein * faktor_b]
            })

            df_melted = df_compare.melt(id_vars=["Nutrisi (per 100g)"], var_name="Produk", value_name="Kandungan")

            fig_compare_bar = px.bar(
                df_melted, x="Nutrisi (per 100g)", y="Kandungan", color="Produk", barmode="group",
                title=f"Perbandingan Nutrisi: {p_a_name} vs {p_b_name} (Distandarisasi per 100g)",
                text_auto='.1f'
            )
            fig_compare_bar.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_compare_bar, use_container_width=True)


elif app_mode == "Riwayat Analisis":
    st.header("9. Riwayat dan Monitoring Konsumsi")
    st.info("Berikut adalah riwayat analisis produk yang pernah Anda periksa.")

    if len(st.session_state.scan_history) == 0:
        st.write("Belum ada riwayat analisis.")
    else:
        history_df = pd.DataFrame(st.session_state.scan_history)

        # Categorize risk scores for visualization
        def categorize_risk(score):
            if score > 75: return "Sangat Tinggi"
            elif score > 50: return "Tinggi"
            elif score > 25: return "Sedang"
            else: return "Rendah"

        history_df["Kategori Risiko"] = history_df["risk_score"].apply(categorize_risk)

        # 1. Tabel Riwayat
        st.subheader("Data Riwayat")
        st.dataframe(history_df[["date", "product_name", "risk_score", "Kategori Risiko", "profile"]].style.format({"risk_score": "{:.2f}%"}))

        st.markdown("---")

        # 2. Visualisasi Dashboard
        st.subheader("Dashboard Analisis Riwayat")

        col1, col2 = st.columns(2)

        with col1:
            # Pie Chart - Proporsi Kategori Risiko
            fig_pie = px.pie(
                history_df,
                names="Kategori Risiko",
                title="Proporsi Kategori Risiko Produk",
                color="Kategori Risiko",
                color_discrete_map={
                    "Sangat Tinggi": "darkred",
                    "Tinggi": "red",
                    "Sedang": "orange",
                    "Rendah": "green"
                },
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Line Chart - Tren Skor Risiko
            fig_line = px.line(
                history_df,
                x="date",
                y="risk_score",
                title="Tren Skor Risiko Konsumsi Seiring Waktu",
                markers=True,
                hover_data=["product_name"]
            )
            fig_line.add_hrect(y0=0, y1=25, line_width=0, fillcolor="green", opacity=0.1)
            fig_line.add_hrect(y0=25, y1=50, line_width=0, fillcolor="orange", opacity=0.1)
            fig_line.add_hrect(y0=50, y1=75, line_width=0, fillcolor="red", opacity=0.1)
            fig_line.add_hrect(y0=75, y1=100, line_width=0, fillcolor="darkred", opacity=0.1)
            fig_line.update_yaxes(title="Skor Risiko (%)", range=[0, 100])
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        # 3. Scatter Plot Kompleks - Korelasi Gula dan Natrium vs Risiko
        st.subheader("Korelasi Kandungan Nutrisi Utama dan Risiko")
        st.info("Visualisasi ini memetakan kadar Gula dan Natrium produk yang pernah Anda periksa. Ukuran gelembung mewakili Skor Risiko.")

        # Ekstrak data nutrisi untuk plotting
        sugar_data = [item.get("gula", 0) for item in history_df["nutrition"]]
        sodium_data = [item.get("natrium", 0) for item in history_df["nutrition"]]

        scatter_df = pd.DataFrame({
            "Produk": history_df["product_name"],
            "Gula (g)": sugar_data,
            "Natrium (mg)": sodium_data,
            "Skor Risiko": history_df["risk_score"],
            "Kategori": history_df["Kategori Risiko"]
        })

        fig_scatter = px.scatter(
            scatter_df,
            x="Gula (g)",
            y="Natrium (mg)",
            size="Skor Risiko",
            color="Kategori",
            hover_name="Produk",
            title="Peta Risiko Berdasarkan Kandungan Gula dan Natrium",
            size_max=30,
            color_discrete_map={
                "Sangat Tinggi": "darkred",
                "Tinggi": "red",
                "Sedang": "orange",
                "Rendah": "green"
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        if st.button("Hapus Riwayat", type="secondary"):
            st.session_state.scan_history = []
            st.rerun()


elif app_mode == "Edukasi Gizi":
    st.header("10. Edukasi dan Rekomendasi Nutrisi Cerdas")

    st.markdown("### Batas Konsumsi Gizi Harian (Kemenkes RI)")
    st.info("Pedoman umum konsumsi gula, garam, dan lemak (G4G1L5) per hari untuk dewasa:")
    st.write("- **Gula:** 4 sendok makan (50 gram)")
    st.write("- **Garam:** 1 sendok teh (5 gram / 2000 mg Natrium)")
    st.write("- **Lemak:** 5 sendok makan (67 gram)")

    st.markdown("---")
    st.markdown("### Membaca Label Informasi Nilai Gizi")
    st.write("Perhatikan hal-hal berikut saat membaca label kemasan:")
    st.write("1. **Takaran Saji**: Semua nilai nutrisi yang tercantum biasanya berdasarkan satu takaran saji, bukan satu kemasan penuh.")
    st.write("2. **Kalori Total**: Perhatikan total kalori per sajian, terutama jika Anda sedang mengatur berat badan.")
    st.write("3. **Natrium/Garam**: Banyak produk camilan dan minuman kemasan menyembunyikan kadar natrium yang sangat tinggi.")

    st.markdown("---")
    st.markdown("### Alternatif Makanan Sehat")
    st.write("- **Ganti Minuman Manis**: Gunakan air putih, teh tawar, atau air infus buah.")
    st.write("- **Camilan Sehat**: Pilih buah potong, kacang edamame, atau yogurt tawar dibandingkan keripik kemasan.")
    st.write("- **Perbanyak Serat**: Konsumsi lebih banyak sayur dan biji-bijian utuh.")


elif app_mode == "Simulasi Konsumsi":
    st.header("8. Simulasi Konsumsi Produk")
    st.info("Masukkan detail produk dan perkirakan dampak risikonya berdasarkan frekuensi konsumsi Anda.")

    st.subheader("Langkah 1: Definisikan Produk")
    product_name = st.text_input("Nama Produk", "Minuman Soda")

    # Tambah input Takaran Saji untuk konsistensi meskipun tidak dipakai kalkulasi total serving di ML
    c0, c1, c2 = st.columns(3)
    takaran_saji = c0.number_input("Takaran Saji (g/ml)", min_value=1.0, value=250.0, format="%.1f")
    energi = c1.number_input("Energi (kkal)", min_value=0, value=150)
    lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=0.0, format="%.1f")

    c3, c4, c5 = st.columns(3)
    lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=0.0, format="%.1f")
    protein = c4.number_input("Protein (g)", min_value=0.0, value=0.0, format="%.1f")
    karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=40.0, format="%.1f")

    c6, c7, c8, c9 = st.columns(4)
    gula = c6.number_input("Gula (g)", min_value=0.0, value=39.0, format="%.1f")
    garam = c7.number_input("Garam (g)", min_value=0.0, value=0.1, format="%.2f")
    natrium = c8.number_input("Natrium (mg)", min_value=0, value=45)
    natrium_benzoat = c9.number_input("Natrium Benzoat (mg)", min_value=0.0, value=0.0, format="%.2f")

    komposisi = st.text_area("Komposisi / Ingredients", "Air Berkarbonasi, Gula, Sirup Fruktosa, Perisa Sintetik, Pengatur Keasaman.")

    st.markdown("---")
    st.subheader("Langkah 2: Atur Pola Konsumsi")
    freq_col, period_col = st.columns(2)
    with freq_col:
        frequency_per_week = st.number_input("Frekuensi konsumsi per minggu (kali/sajian)", min_value=1, value=3)
    with period_col:
        simulation_period_months = st.selectbox("Periode Simulasi (Bulan)", [1, 3, 6, 12])

    st.markdown("---")
    simulation_button = st.button("📈 Jalankan Simulasi", type="primary")

    if simulation_button:
        with st.spinner("Menjalankan simulasi konsumsi..."):
            nutrition_data = {
                'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                'garam': garam, 'natrium': natrium, 'natrium_benzoat': natrium_benzoat
            }
            risk_score, _, _ = analyze_product_fully(
                nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
            )

            st.subheader(f"Hasil Analisis Dasar untuk '{product_name}'")
            res_col, _ = st.columns(2)
            with res_col:
                st.metric(label="Skor Risiko per 1x Konsumsi", value=f"{risk_score:.2f}%")
                if risk_score > 75: st.error("Risiko Sangat Tinggi")
                elif risk_score > 50: st.warning("Risiko Tinggi")
                elif risk_score > 25: st.warning("Risiko Sedang")
                else: st.success("Risiko Rendah")

            st.markdown("---")
            st.subheader(f"Simulasi Akumulasi Selama {simulation_period_months} Bulan")

            profile_daily_limits = current_threshold

            days_in_period = simulation_period_months * 30.44
            weeks_in_period = days_in_period / 7
            total_servings = frequency_per_week * weeks_in_period

            total_gula = gula * total_servings
            total_natrium = natrium * total_servings
            total_lemak_jenuh = lemak_jenuh * total_servings

            limit_gula = profile_daily_limits['gula'] * days_in_period
            limit_natrium = profile_daily_limits['natrium'] * days_in_period
            limit_lemak_jenuh = profile_daily_limits['lemak_jenuh'] * days_in_period

            st.write(f"Dengan mengonsumsi **{product_name}** sebanyak **{frequency_per_week}** sajian seminggu, estimasi asupan Anda dari produk ini saja adalah:")

            percent_gula = (total_gula / limit_gula) * 100 if limit_gula > 0 else 0
            st.write(f"**Gula**: **{total_gula:.1f}g** / {limit_gula:.1f}g dari batas maksimal periode.")
            st.progress(min(int(percent_gula), 100))
            if percent_gula > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_gula - 100:.0f}%** dari batas aman gula Anda.")
            elif percent_gula > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_gula:.0f}%** dari alokasi gula Anda untuk periode ini.")

            percent_natrium = (total_natrium / limit_natrium) * 100 if limit_natrium > 0 else 0
            st.write(f"**Natrium**: **{total_natrium / 1000:.2f}g** / {limit_natrium / 1000:.2f}g dari batas maksimal periode.")
            st.progress(min(int(percent_natrium), 100))
            if percent_natrium > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_natrium - 100:.0f}%** dari batas aman natrium Anda.")
            elif percent_natrium > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_natrium:.0f}%** dari alokasi natrium Anda untuk periode ini.")

            percent_lemak_jenuh = (total_lemak_jenuh / limit_lemak_jenuh) * 100 if limit_lemak_jenuh > 0 else 0
            st.write(f"**Lemak Jenuh**: **{total_lemak_jenuh:.1f}g** / {limit_lemak_jenuh:.1f}g dari batas maksimal periode.")
            st.progress(min(int(percent_lemak_jenuh), 100))
            if percent_lemak_jenuh > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_lemak_jenuh - 100:.0f}%** dari batas aman lemak jenuh Anda.")
            elif percent_lemak_jenuh > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_lemak_jenuh:.0f}%** dari alokasi lemak jenuh Anda untuk periode ini.")

            st.caption(f"Perhitungan berdasarkan profil '{user_profile}' selama {simulation_period_months} bulan. Ingat, ini baru dari 1 produk, belum memperhitungkan asupan makanan berat Anda sehari-hari.")
