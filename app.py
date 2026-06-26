from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# 1. PENTING: Beri tahu Flask bahwa file HTML dan CSS ada di folder 'dist'
app = Flask(__name__, 
            static_folder='dist/assets', 
            static_url_path='/assets', 
            template_folder='dist')

model = joblib.load('ai_model.pkl')

# 2. Routing untuk memunculkan halaman web dari Figma
@app.route('/')
def home():
    return render_template('index.html')

# 3. Routing API untuk proses Prediksi (Pastikan ini sesuai dengan kodinganmu)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame([data])
    # ..(Proses data pakai scaler)..
    prediction = model.predict(df_input)[0]
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)