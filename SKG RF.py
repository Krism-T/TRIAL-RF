import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from collections import Counter

# Library Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

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
    """Mendownload file MAPPING.xlsx dari Google Sheets"""
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
    """
    Melakukan training model sekali saja saat server restart.
    Hasilnya disimpan di cache agar aplikasi cepat.
    """
    with st.spinner('Sedang melatih model (ini hanya terjadi sekali saat start)...'):
        # 1. Load Data
        sheet_url = 'https://docs.google.com/spreadsheets/d/1AfcS9SYlAba88BgWKhOIyKNR8TMlh379wvi8RlklVCI/edit?usp=sharing'
        csv_url = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')
        df = pd.read_csv(csv_url).dropna()
        
        # 2. Preprocessing
        TARGET_COLUMN = 'LevelKompetensi'
        feature_cols = [c for c in df.columns if c not in [TARGET_COLUMN, 'ID']]
        X = df[feature_cols].apply(pd.to_numeric, errors='raise')
        
        le = LabelEncoder()
        y = le.fit_transform(df[TARGET_COLUMN])
        
        # SMOTE Setup
        target_smote_labels = ['Novice Teacher', 'Expert Teacher']
        # Filter label yang ada di dataset
        existing_labels = set(le.classes_)
        valid_smote_labels = [lbl for lbl in target_smote_labels if lbl in existing_labels]
        
        smote_classes = le.transform(valid_smote_labels)
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Hitung target count untuk SMOTE
        train_class_counts = Counter(y_train)
        majority_class_code = train_class_counts.most_common(1)[0][0]
        target_count_for_minority = train_class_counts[majority_class_code]

        # Strategi SMOTE Custom
        def smote_strategy(y_array):
            counter = Counter(y_array)
            strategy = {}
            # Oversample kelas target
            for cls_code in smote_classes:
                if cls_code in counter:
                    strategy[cls_code] = target_count_for_minority if counter[cls_code] < target_count_for_minority else counter[cls_code]
            # Pertahankan kelas lain
            for cls_code, count in counter.items():
                if cls_code not in smote_classes:
                    strategy[cls_code] = count
            return strategy

        # Pipeline
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=1, sampling_strategy=smote_strategy)),
            ('rf', RandomForestClassifier(random_state=42))
        ])

        # GridSearch (Disederhanakan agar tidak timeout di Cloud)
        param_grid = {
            'rf__n_estimators': [100], 
            'rf__max_depth': [10, None],
            'rf__min_samples_leaf': [1]
        }
        
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluasi Singkat
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        
        # Simpan fitur yang digunakan model
        model_features = X.columns.tolist()
        
        return best_model, le, model_features, acc, report, confusion_matrix(y_test, y_pred)

# ==========================================
# FUNGSI 3: LOAD MAPPING DARI EXCEL
# ==========================================
def load_mapping_dict(path):
    df_map = pd.read_excel(path)
    mapping = {}
    for var in df_map["Variabel"].unique():
        sub = df_map[df_map["Variabel"] == var]
        mapping[var] = {str(row["Meaning"]): int(row["Code"]) for _, row in sub.iterrows()}
    return mapping

# ==========================================
# MAIN APP (LOGIKA UTAMA)
# ==========================================
def main():
    st.title("👨‍🏫 Prediksi Level Kompetensi Guru (SKG 360)")
    
    # 1. Persiapkan File Mapping & Model
    mapping_file = get_mapping_file()
    if not mapping_file:
        st.stop()
        
    MAPPING = load_mapping_dict(mapping_file)
    categorical_features = list(MAPPING.keys())

    # Load Model (akan training otomatis jika belum ada di memori)
    try:
        model, le, model_features, acc, report, cm = train_model()
    except Exception as e:
        st.error(f"Gagal melatih model: {e}")
        st.stop()

    st.success(f"Model siap! Akurasi Test: {acc:.2%}")

    # 2. Form Input
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
            input_strings["TOEFL>400"] = st.selectbox("Memiliki Sertifikat TOEFL ≥ 400?", _options("TOEFL>400"))
            input_strings["Pendidikan"] = st.selectbox("Pendidikan Terakhir", _options("Pendidikan"))
            input_strings["Prodi"] = st.selectbox("Program Studi S1", _options("Prodi"))
            input_strings["UnivS1"] = st.selectbox("Universitas S1", _options("UnivS1"))

        submitted = st.form_submit_button("Prediksi Level Kompetensi")

    # 3. Proses Prediksi
    if submitted:
        raw_data = {"katalog.Usia": usia, "Pengalaman": pengalaman}
        valid_input = True

        for kol, label_str in input_strings.items():
            if label_str == "--Pilih--":
                st.warning(f"Mohon pilih nilai untuk '{kol}'.")
                valid_input = False
            else:
                raw_data[kol] = MAPPING[kol][label_str]

        if valid_input:
            # Bangun dataframe input
            input_df = pd.DataFrame([raw_data])
            # One-Hot Encoding sesuai training
            input_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
            
            # Sesuaikan kolom dengan fitur model (tambah kolom hilang dengan 0)
            final_input = pd.DataFrame(columns=model_features)
            for col in input_encoded.columns:
                if col in final_input.columns:
                    final_input.loc[0, col] = input_encoded[col][0]
            
            final_input = final_input.fillna(0)

            # Prediksi
            try:
                pred_idx = model.predict(final_input)[0]
                pred_label = le.inverse_transform([pred_idx])[0]
                proba = model.predict_proba(final_input)[0]

                st.divider()
                st.markdown(f"### Hasil Prediksi: **{pred_label}**")
                
                prob_df = pd.DataFrame({
                    "Level": le.classes_,
                    "Probabilitas": proba
                }).sort_values("Probabilitas", ascending=False)
                
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

                if "Expert" in pred_label:
                    st.balloons()
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")

    # 4. Tampilkan Metrik Evaluasi (Optional)
    with st.expander("Lihat Detail Performa Model"):
        st.text(report)
        st.write("Confusion Matrix:")
        st.dataframe(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

if __name__ == "__main__":
    main()