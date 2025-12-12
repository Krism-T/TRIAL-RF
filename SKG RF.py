import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import gspread
from google.oauth2.service_account import Credentials
from collections import Counter

# Library Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Imbalanced Learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Prediksi Kompetensi Guru", layout="wide")

# ==========================================
# FUNGSI 1: DOWNLOAD MAPPING (CACHE)
# ==========================================
@st.cache_data
def get_mapping_file():
    sheet_url = 'https://docs.google.com/spreadsheets/d/1K74HlgKj19djz9EUeg0ZUIWMzM9Kt4TvS0oPn_CDsSc/edit?usp=sharing'
    excel_export_url = sheet_url.replace('/edit?usp=sharing', '/export?format=xlsx')
    file_name = 'MAPPING.xlsx'
    try:
        response = requests.get(excel_export_url)
        response.raise_for_status()
        with open(file_name, 'wb') as f:
            f.write(response.content)
        return file_name
    except Exception as e:
        st.error(f"Gagal mendownload MAPPING.xlsx: {e}")
        return None

# ==========================================
# FUNGSI 2: TRAINING MODEL (CACHE RESOURCE)
# ==========================================
@st.cache_resource
def train_model():
    with st.spinner('Sedang melatih model...'):
        sheet_url = 'https://docs.google.com/spreadsheets/d/1AfcS9SYlAba88BgWKhOIyKNR8TMlh379wvi8RlklVCI/edit?usp=sharing'
        csv_url = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')
        df = pd.read_csv(csv_url).dropna()
        
        TARGET_COLUMN = 'LevelKompetensi'
        feature_cols = [c for c in df.columns if c not in [TARGET_COLUMN, 'ID']]
        X = df[feature_cols].apply(pd.to_numeric, errors='raise')
        
        le = LabelEncoder()
        y = le.fit_transform(df[TARGET_COLUMN])
        
        target_smote_labels = ['Novice Teacher', 'Expert Teacher']
        existing_labels = set(le.classes_)
        valid_smote_labels = [lbl for lbl in target_smote_labels if lbl in existing_labels]
        smote_classes = le.transform(valid_smote_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_class_counts = Counter(y_train)
        majority_class_code = train_class_counts.most_common(1)[0][0]
        target_count_for_minority = train_class_counts[majority_class_code]

        def smote_strategy(y_array):
            counter = Counter(y_array)
            strategy = {}
            for cls_code in smote_classes:
                if cls_code in counter:
                    strategy[cls_code] = target_count_for_minority if counter[cls_code] < target_count_for_minority else counter[cls_code]
            for cls_code, count in counter.items():
                if cls_code not in smote_classes:
                    strategy[cls_code] = count
            return strategy

        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=1, sampling_strategy=smote_strategy)),
            ('rf', RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            'rf__n_estimators': [100], 
            'rf__max_depth': [10, None],
            'rf__min_samples_leaf': [1]
        }
        
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        model_features = X.columns.tolist()
        
        return best_model, le, model_features, acc, report, confusion_matrix(y_test, y_pred)

# ==========================================
# FUNGSI 3: SIMPAN KE GOOGLE SHEET
# ==========================================
def simpan_ke_sheet(data_row):
    """
    Mengirim data ke Google Sheet menggunakan Service Account.
    Mengambil credentials dari st.secrets["gcp_service_account"]
    """
    # Cek apakah secrets sudah diatur
    if "gcp_service_account" not in st.secrets:
        st.warning("⚠️ Data tidak disimpan: Secrets 'gcp_service_account' belum diatur di Streamlit Cloud.")
        return

    try:
        # 1. Autentikasi
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=scopes
        )
        gc = gspread.authorize(credentials)

        # 2. Buka Spreadsheet (Ganti URL ini dengan URL Google Sheet 'DATABASE' Anda)
        # SANGAT DISARANKAN BUAT SHEET BARU KHUSUS LOG, JANGAN CAMPUR DENGAN DATA TRAINING
        # Pastikan sheet ini sudah di-SHARE ke email service account
        SHEET_URL = "https://docs.google.com/spreadsheets/d/1WduVmr2BQQU49bki21NcU3v9lDjDzl-nQvHRBd4k-bk/edit?gid=0#gid=0"
        
        sh = gc.open_by_url(SHEET_URL)
        
        # Coba buka tab bernama "Hasil_Prediksi", jika tidak ada, gunakan sheet pertama
        try:
            worksheet = sh.worksheet("Hasil_Prediksi")
        except:
            # Jika tab tidak ada, buat baru (opsional) atau pakai sheet1
            # worksheet = sh.sheet1 
            st.error("Tab 'Hasil_Prediksi' tidak ditemukan di Google Sheet. Mohon buat tab baru dengan nama tersebut.")
            return

        # 3. Append Data
        worksheet.append_row(data_row)
        st.toast("✅ Data berhasil disimpan ke Google Sheet!", icon="💾")
        
    except Exception as e:
        st.error(f"Gagal menyimpan data: {e}")

# ==========================================
# FUNGSI 4: LOAD MAPPING
# ==========================================
def load_mapping_dict(path):
    df_map = pd.read_excel(path)
    mapping = {}
    for var in df_map["Variabel"].unique():
        sub = df_map[df_map["Variabel"] == var]
        mapping[var] = {str(row["Meaning"]): int(row["Code"]) for _, row in sub.iterrows()}
    return mapping

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.title("👨‍🏫 Prediksi Level Kompetensi Guru (SKG 360)")
    st.markdown("""
    Aplikasi ini menggunakan model **Random Forest** untuk memprediksi level kompetensi guru 
    berdasarkan data demografis/latar belakang guru.
    """)
    
    mapping_file = get_mapping_file()
    if not mapping_file:
        st.stop()
    MAPPING = load_mapping_dict(mapping_file)
    categorical_features = list(MAPPING.keys())

    try:
        model, le, model_features, acc, report, cm = train_model()
    except Exception as e:
        st.error(f"Gagal melatih model: {e}")
        st.stop()

    st.success(f"Model siap! Akurasi Test: {acc:.2%}")

    # Form Input
    st.markdown("---")
    if 'form_key_counter' not in st.session_state:
        st.session_state.form_key_counter = 0

    def reset_form():
        st.session_state.form_key_counter += 1

    st.button("Input Baru", on_click=reset_form)
    
    def _options(var_name):
        opts = list(MAPPING.get(var_name, {}).keys())
        return ["--Pilih--"] + opts if opts else ["--Pilih--"]

    with st.form(key=f"my_form_{st.session_state.form_key_counter}"):
        st.subheader("Input Data Guru")
        
        # INPUT NAMA GURU (BARU)
        nama_guru = st.text_input("Nama Guru / ID Guru", placeholder="Contoh: Budi Santoso (ID-12345)")
        
        col1, col2 = st.columns(2)
        input_strings = {}
        
        with col1:
            usia = st.number_input("Usia (tahun)", min_value=20, max_value=70, value=30)
            input_strings["Sekolah"] = st.selectbox("Sekolah", _options("Sekolah"))
            input_strings["Departemen"] = st.selectbox("Departemen", _options("Departemen"))
            input_strings["Gender"] = st.selectbox("Gender", _options("Gender"))
            input_strings["Mapel"] = st.selectbox("Mata Pelajaran", _options("Mapel"))

        with col2:
            pengalaman = st.number_input("Pengalaman Mengajar (tahun)", min_value=0, max_value=40, value=5)
            input_strings["Region Lahir"] = st.selectbox("Region Lahir", _options("Region Lahir"))
            input_strings["katalog.StatusKaryawanChoices"] = st.selectbox("Status Karyawan", _options("katalog.StatusKaryawanChoices"))
            input_strings["katalog.MaritalStatus"] = st.selectbox("Status Perkawinan", _options("katalog.MaritalStatus"))
            input_strings["TOEFL>400"] = st.selectbox("Memiliki sertifikat TOEFL/EAT/IELTS ?", _options("TOEFL>400"))
            input_strings["Pendidikan"] = st.selectbox("Pendidikan Terakhir", _options("Pendidikan"))
            input_strings["Prodi"] = st.selectbox("Program Studi S1", _options("Prodi"))
            input_strings["UnivS1"] = st.selectbox("Universitas S1", _options("UnivS1"))

        submitted = st.form_submit_button("Prediksi & Simpan")

    if submitted:
        if not nama_guru:
            st.error("Mohon isi Nama/ID Guru terlebih dahulu.")
        else:
            raw_data = {"katalog.Usia": usia, "Pengalaman": pengalaman}
            valid_input = True
            
            # List untuk menyimpan data mentah (untuk dikirim ke Excel)
            # Urutan harus sesuai dengan header di Google Sheet "Hasil_Prediksi"
            data_to_save = [nama_guru, usia, pengalaman] 

            for kol, label_str in input_strings.items():
                if label_str == "--Pilih--":
                    st.warning(f"Mohon pilih nilai untuk '{kol}'.")
                    valid_input = False
                else:
                    raw_data[kol] = MAPPING[kol][label_str]
                    data_to_save.append(label_str) # Simpan teks aslinya ke Excel

            if valid_input:
                # Preprocessing & Prediksi
                input_df = pd.DataFrame([raw_data])
                input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
                final_input = pd.DataFrame(columns=model_features)
                for col in input_encoded.columns:
                    if col in final_input.columns:
                        final_input.loc[0, col] = input_encoded[col][0]
                final_input = final_input.fillna(0)

                try:
                    pred_idx = model.predict(final_input)[0]
                    pred_label = le.inverse_transform([pred_idx])[0]
                    proba = model.predict_proba(final_input)[0]

                    st.divider()
                    st.markdown(f"### Hasil Prediksi: **{pred_label}**")
                    
                    # Tambahkan hasil prediksi ke data yang mau disimpan
                    data_to_save.append(pred_label)
                    
                    # === SIMPAN KE GOOGLE SHEET ===
                    simpan_ke_sheet(data_to_save)
                    # ==============================

                    if "Expert" in pred_label:
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"Error saat prediksi: {e}")

if __name__ == "__main__":
    main()