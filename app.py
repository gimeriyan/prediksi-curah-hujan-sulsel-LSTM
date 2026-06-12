from flask import Flask, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import logging

# Menonaktifkan log tensorflow yang mengganggu di terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Pastikan nama file CSV sesuai dengan yang ada di folder luar
DATA_PATH = os.path.join(BASE_DIR, 'CH_Master_10Tahun_2015_2024.csv')

# =========================================================
# 1. KONFIGURASI KAMUS DATA UNTUK 3 STASIUN
# =========================================================
# PENTING: Pastikan nama file .h5 dan .pkl di bawah ini SAMA PERSIS 
# dengan nama file yang ada di dalam folder 'Model_Tersimpan'.
# Pastikan juga 'csv_name' SAMA PERSIS dengan teks di Excel/CSV Anda.
STASIUN_CONFIG = {
    'hasanuddin': {
        'model_name': 'model_harian_Stamet Hasanuddin.h5',
        'scaler_name': 'scaler_harian_Stamet Hasanuddin.pkl',
        'csv_name': 'Stamet Hasanuddin'
    },
    'pongtiku': {
        'model_name': 'model_harian_Stamet Pongtiku Tana Toraja.h5',
        'scaler_name': 'scaler_harian_Stamet Pongtiku Tana Toraja.pkl',
        'csv_name': 'Stamet Pongtiku Tana Toraja'
    },
    'benteng': {
        'model_name': 'model_harian_Benteng_Bontoharu.h5',          
        'scaler_name': 'scaler_harian_Benteng_Bontoharu.pkl',       
        'csv_name': 'Benteng / Bontoharu'                               
    }
}

# =========================================================
# 2. LOAD SEMUA MODEL & SCALER KE MEMORI RAM
# =========================================================
models = {}
scalers = {}

print("Memuat semua model dan scaler ke dalam memori server...")
for stasiun_id, config in STASIUN_CONFIG.items():
    model_path = os.path.join(BASE_DIR, 'Model_Tersimpan', config['model_name'])
    scaler_path = os.path.join(BASE_DIR, 'Model_Tersimpan', config['scaler_name'])
    
    try:
        models[stasiun_id] = load_model(model_path)
        scalers[stasiun_id] = joblib.load(scaler_path)
        print(f"[OK] Model & Scaler {stasiun_id} berhasil dimuat.")
    except Exception as e:
        print(f"[PERINGATAN] Gagal memuat {stasiun_id}. Pastikan file ada di folder Model_Tersimpan. Error: {e}")

# =========================================================
# 3. ROUTING WEB UTAMA
# =========================================================
@app.route('/')
def home():
    return render_template('index.html')

# =========================================================
# 4. ENDPOINT API DINAMIS (BISA MELAYANI 3 STASIUN)
# =========================================================
@app.route('/api/prediksi/<id_stasiun>', methods=['GET'])
def get_prediksi(id_stasiun):
    # Cek apakah stasiun yang diminta website ada di konfigurasi kita
    if id_stasiun not in STASIUN_CONFIG:
        return jsonify({'status': 'error', 'message': 'Stasiun tidak valid!'}), 404
        
    try:
        # Ambil konfigurasi, model, dan scaler spesifik untuk stasiun ini
        config = STASIUN_CONFIG[id_stasiun]
        model = models[id_stasiun]
        scaler = scalers[id_stasiun]
        
        # Baca dataset dan filter berdasarkan nama stasiun di CSV
        df = pd.read_csv(DATA_PATH, delimiter=';')
        df_stasiun = df[df['NAME'] == config['csv_name']].copy()
        
        if df_stasiun.empty:
            return jsonify({'status': 'error', 'message': f'Data untuk {config["csv_name"]} tidak ditemukan di CSV!'}), 404

        # --- PRA-PEMROSESAN (Standar Meteorologi WMO) ---
        df_stasiun['RAINFALL DAY MM'] = df_stasiun['RAINFALL DAY MM'].replace(8888.0, 0.0)
        df_stasiun['RAINFALL DAY MM'] = df_stasiun['RAINFALL DAY MM'].replace(9999.0, np.nan)
        df_stasiun['RAINFALL DAY MM'] = df_stasiun['RAINFALL DAY MM'].interpolate(method='linear').fillna(0.0)
        
        # --- PROSES MACHINE LEARNING ---
        # 1. Ambil 30 hari historis terakhir
        data_terakhir = df_stasiun['RAINFALL DAY MM'].values[-30:]
        
        # 2. Skalakan data (0-1) menggunakan scaler masing-masing
        data_scaled = scaler.transform(data_terakhir.reshape(-1, 1))
        
        # 3. Ubah bentuknya menjadi 3D Array untuk masuk ke LSTM
        X_input = np.reshape(data_scaled, (1, 30, 1))
        
        # 4. Model melakukan inferensi (menebak 30 hari ke depan)
        prediksi_scaled = model.predict(X_input)
        
        # 5. Kembalikan tebakan ke skala milimeter (mm) asli
        prediksi_asli = scaler.inverse_transform(prediksi_scaled)[0]
        
        # 6. Bersihkan tebakan minus (jadi 0) dan bulatkan 1 angka di belakang koma
        prediksi_list = [round(max(0, float(val)), 1) for val in prediksi_asli]
        
        # Kirimkan data hasil prediksi dalam bentuk JSON ke Website
        return jsonify({
            'status': 'success',
            'stasiun': config['csv_name'],
            'prediksi': prediksi_list
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)