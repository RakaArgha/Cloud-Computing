# app.py - Microservice & Web Dashboard untuk Predictive Maintenance
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model (Gunakan try-except agar tidak error jika dijalankan tanpa model saat testing awal)
try:
    model = joblib.load('ai_model.pkl')
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    model_ready = True
except:
    model_ready = False

# ==========================================
# 1. FRONTEND: TAMPILAN WEB UNTUK DOSEN
# ==========================================
@app.route('/', methods=['GET'])
def home():
    # Ini adalah kode HTML sederhana yang akan ditampilkan di Web Browser
    html_page = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IoT Predictive Maintenance</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }
            .container { max-width: 600px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0px 0px 10px #ccc; }
            input, select { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; }
            button { background-color: #28a745; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; width: 100%; }
            button:hover { background-color: #218838; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🤖 Dashboard Cloud AI: Monitor Mesin Robot</h2>
            <p>Masukkan data log sensor dari mesin AMR untuk diprediksi oleh AI di Cloud.</p>
            <form action="/predict-web" method="POST">
                <label>Suhu Udara (K):</label>
                <input type="number" step="any" name="air_temp" required value="298.1">
                
                <label>Suhu Proses Mesin (K):</label>
                <input type="number" step="any" name="process_temp" required value="308.6">
                
                <label>Kecepatan Rotasi (rpm):</label>
                <input type="number" name="rotational_speed" required value="1551">
                
                <label>Torsi (Nm):</label>
                <input type="number" step="any" name="torque" required value="42.8">
                
                <label>Keausan Alat / Tool Wear (min):</label>
                <input type="number" name="tool_wear" required value="0">
                
                <label>Tipe Kualitas Mesin:</label>
                <select name="machine_type">
                    <option value="L">L (Low)</option>
                    <option value="M">M (Medium)</option>
                    <option value="H">H (High)</option>
                </select>
                
                <button type="submit">Prediksi Kerusakan Mesin</button>
            </form>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_page)

# ==========================================
# 2. BACKEND: LOGIKA PREDIKSI SAAT TOMBOL DIKLIK
# ==========================================
@app.route('/predict-web', methods=['POST'])
def predict_web():
    if not model_ready:
        return "<h1>Error: File Model AI (.pkl) tidak ditemukan di server Cloud!</h1>"
        
    # Mengambil data yang diketik di web
    data = request.form
    
    # Encoding manual untuk Tipe Mesin (Sesuai dengan saat training)
    type_L = 1 if data['machine_type'] == 'L' else 0
    type_M = 1 if data['machine_type'] == 'M' else 0

    # Menyusun fitur
    features = np.array([[
        float(data['air_temp']), float(data['process_temp']), 
        float(data['rotational_speed']), float(data['torque']), 
        float(data['tool_wear']), type_L, type_M
    ]])
    
    # Preprocessing (Scaler & PCA) -> Model Predict
    scaled_features = scaler.transform(features)
    pca_features = pca.transform(scaled_features)
    prediction = model.predict(pca_features)
    
    # Hasil
    if prediction[0] == 1:
        hasil = "<h1 style='color:red;'>⚠️ PERINGATAN: MESIN BERISIKO RUSAK! (Failure)</h1>"
    else:
        hasil = "<h1 style='color:green;'>✅ STATUS MESIN: NORMAL (No Failure)</h1>"
        
    return f"<div style='font-family: Arial; text-align: center; margin-top: 50px;'>{hasil}<br><br><a href='/'>Kembali ke Dashboard</a></div>"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)