import os
import joblib
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global instances
MODEL_PATH = 'disease_model.pkl'
PROCESSED_DATA_PATH = 'processed_dataset.csv'
MEDQUAD_PATH = 'medDataset_processed.csv'

model = None
disease_database = {}
medquad_data = None
IMAGE_MODEL = None

def get_image_model():
    global IMAGE_MODEL
    if IMAGE_MODEL is None:
        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            IMAGE_MODEL = MobileNetV2(weights='imagenet')
            logger.info("Vision Engine Loaded")
        except Exception as e:
            logger.error(f"Vision Engine Error: {e}")
    return IMAGE_MODEL

def load_model_and_data():
    global model, disease_database, medquad_data
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except: pass

    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH)
            for _, row in df.iterrows():
                disease_database[str(row['disease_name']).lower()] = {
                    'tablets': row.get('tablets', 'Consult a doctor'),
                    'precautions': row.get('precautions', 'Rest and hydrate'),
                    'description': row.get('Description', 'No details available.')
                }
        except: pass

    if os.path.exists(MEDQUAD_PATH):
        try:
            medquad_data = pd.read_csv(MEDQUAD_PATH).head(5000)
        except: pass

load_model_and_data()

def detect_language(text):
    keywords = ['mujhe', 'hai', 'ho', 'raha', 'rahi', 'dard', 'kya', 'karun', 'bukhar', 'sardi']
    return "hinglish" if any(k in text.lower() for k in keywords) else "english"

import re

def handle_greetings(query):
    greetings = {"hi": "Hello!", "hello": "Hi!", "hey": "Hey!", "good morning": "Good morning!"}
    
    # If the user asks a long medical question that happens to contain "hi", we should probably answer the medical part.
    # But to be safe, we check if the greeting is a distinct word.
    for k, v in greetings.items():
        if re.search(r'\b' + re.escape(k) + r'\b', query.lower()):
            # If it's a long query (more than 5 words), skip greeting and go to prediction
            if len(query.split()) > 5:
                return None
            return f"{v} I'm HealthMate AI. How can I help?"
    return None

@app.route('/predict-image', methods=['POST'])
def predict_image():
    img_model = get_image_model()
    if img_model is None or 'image' not in request.files:
        return jsonify({'error': 'Vision engine unavailable'}), 400
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
        img = Image.open(BytesIO(request.files['image'].read())).convert('RGB').resize((224, 224))
        x = preprocess_input(np.expand_dims(np.array(img), axis=0))
        preds = img_model.predict(x)
        res = decode_predictions(preds, top=1)[0][0][1].replace('_', ' ')
        return jsonify({
            'formatted_response': f"Analysis Result: I detected {res.upper()} in the image.\n\nPlease consult a specialist for medical confirmation.",
            'source': 'Vision AI'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        data = request.get_json()
        query = data.get('symptoms', '').strip()
        if not query: return jsonify({'error': 'Empty query'}), 400
        
        greet = handle_greetings(query)
        if greet: return jsonify({'formatted_response': f"👋 {greet}", 'source': 'Chat Engine'}), 200

        expert = None
        if medquad_data is not None:
            query_words = set(query.lower().split())
            if len(query_words) >= 2:
                for _, row in medquad_data.iterrows():
                    q_words = set(str(row['Question']).lower().split())
                    if len(query_words.intersection(q_words)) >= 3:
                        expert = row['Answer']
                        break
        
        if expert: return jsonify({'formatted_response': f"Expert Knowledge:\n\n{expert}", 'source': 'MedQuAD'}), 200

        if model:
            probs = model.predict_proba([query])[0]
            max_prob = max(probs)
            pred = model.classes_[probs.argmax()]
            
            lang = detect_language(query)
            
            # Confidence Threshold - if the model is guessing blindly
            if max_prob < 0.10:
                if lang == "hinglish":
                    res = "Maaf kijiye, mujhe in lakshano (symptoms) se bimari theek se samajh nahi aa rahi. Kripya thoda aur detail mein batayein (jaise: fever, cough, sar dard)."
                else:
                    res = "I'm not completely sure based on those symptoms. Could you please provide a bit more detail (e.g., fever, cough, headache)?"
                return jsonify({'formatted_response': res, 'source': 'ML Engine'}), 200

            info = disease_database.get(pred.lower(), {
                'tablets': 'Consult a doctor',
                'precautions': 'Take rest',
                'description': 'No additional details found.'
            })
            
            if lang == "hinglish":
                res = f"Aapke lakshano se lagta hai ki aapko {pred.upper()} ho sakta hai.\n\n💊 Tablets: {info['tablets']}\n🛡️ Precautions: {info['precautions']}"
            else:
                res = f"Based on your symptoms, you might have {pred.upper()}.\n\nℹ️ Details: {info['description']}\n💊 Tablets: {info['tablets']}\n🛡️ Precautions: {info['precautions']}"
            
            return jsonify({'disease': pred, 'formatted_response': res, 'source': 'ML Engine'}), 200

        return jsonify({'error': 'AI engine offline'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready"}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)



