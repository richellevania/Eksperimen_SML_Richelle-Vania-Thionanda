# preprocessing/automate_Richelle Vania Thionanda.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
import os
import sys

# --- KONFIGURASI PATHS DEFAULT ---
# Asumsi: Skrip ini dijalankan dari folder 'preprocessing/'
# Data mentah ada satu level di atas (..)
DEFAULT_RAW_PATH = "../breastcancer_raw.csv" 
DEFAULT_PROCESSED_DIR = "breastcancer_preprocessing"

def preprocess_breastcancer(
    raw_data_path: str = DEFAULT_RAW_PATH, 
    processed_data_dir: str = DEFAULT_PROCESSED_DIR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Melakukan preprocessing otomatis pada dataset Breast Cancer Wisconsin (Diagnostic).
    
    Langkah-langkah:
    1. Membaca file CSV mentah
    2. Menghapus kolom 'id' dan 'Unnamed: 32' (jika ada)
    3. Meng-encode target 'diagnosis' ('M'=1, 'B'=0)
    4. Imputasi missing values dengan median (untuk konsistensi MLOps)
    5. Train/Test Split (80/20, stratify)
    6. Standarisasi fitur numerik (StandardScaler)
    7. Menyimpan data latih & uji yang sudah diproses ke CSV
    8. Mengembalikan 4 DataFrame/Series (X_train, X_test, y_train, y_test)
    """
    
    print(f"Memulai preprocessing dari: {raw_data_path}")
    
    # 1. Data Loading
    try:
        # Cek lokasi file relatif terhadap skrip (untuk fleksibilitas)
        if not os.path.isfile(raw_data_path):
             # Jika tidak ditemukan, cek di parent folder
             raw_data_path = os.path.join(os.path.dirname(__file__), '..', os.path.basename(raw_data_path))
             
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {raw_data_path}. Pastikan path benar.")
        sys.exit(1)
        
    print(f"Data mentah dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    
    # 2. Drop Kolom Tidak Relevan ('id' dan 'Unnamed: 32')
    df = df.drop(columns=['id'], errors='ignore') # Hapus 'id'
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Hapus 'Unnamed: xx'
    print("Kolom 'id' dan 'Unnamed: 32' (jika ada) telah dihapus.")

    # 3. Encoding Target ('diagnosis': M=1, B=0)
    try:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    except KeyError:
        print("Error: Kolom 'diagnosis' tidak ditemukan.")
        sys.exit(1)
    
    # 4. Pemisahan Fitur (X) dan Target (y)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    feature_columns = X.columns # Simpan nama kolom sebelum transformasi

    # 5. Penanganan Missing Values (Imputasi Median)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_columns)
    print("Missing values diimputasi dengan median.")


    # 6. Train/Test Split (80% Latih, 20% Uji)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data dibagi: X_train={X_train.shape}, X_test={X_test.shape}")

    # 7. Penskalaan Fitur (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Penskalaan fitur (StandardScaler) berhasil.")

    # 8. Ubah kembali ke DataFrame (menggunakan nama kolom yang disimpan)
    X_train = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_columns)
    
    # 9. Simpan Data yang Telah Diproses
    os.makedirs(processed_data_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(processed_data_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_data_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_data_dir, "y_test.csv"), index=False)
    
    print(f"\n✅ Preprocessing selesai. Data siap dilatih disimpan di folder: {processed_data_dir}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Fungsi akan membuat folder dan menyimpan file saat dijalankan
    X_train, X_test, y_train, y_test = preprocess_breastcancer()