import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_and_merge_data():
    """Merges multiple CSV files into a single training dataset."""
    try:
        # 1. Load the main dataset (Symptoms and Diseases)
        if not os.path.exists('dataset.csv'):
            logger.error("'dataset.csv' not found. Please ensure all Kaggle files are in the folder.")
            return None
            
        df_raw = pd.read_csv('dataset.csv')
        
        # Combine all symptom columns (Symptom_1 to Symptom_17) into a single string
        symptom_cols = [c for c in df_raw.columns if 'Symptom' in c]
        # We replace underscores with spaces to help the Vectorizer
        df_raw['symptoms'] = df_raw[symptom_cols].apply(
            lambda x: ' '.join([str(val).replace('_', ' ').strip() for val in x if pd.notna(val) and str(val).lower() != 'nan']), axis=1
        )
        
        # Keep only necessary columns
        df = df_raw[['Disease', 'symptoms']].copy()
        df.rename(columns={'Disease': 'disease_name'}, inplace=True)
        # Strip whitespace from disease names
        df['disease_name'] = df['disease_name'].str.strip()

        # 2. Merge with Descriptions
        if os.path.exists('symptom_Description.csv'):
            df_desc = pd.read_csv('symptom_Description.csv')
            df_desc.rename(columns={'Disease': 'disease_name'}, inplace=True)
            df_desc['disease_name'] = df_desc['disease_name'].str.strip()
            df = pd.merge(df, df_desc, on='disease_name', how='left')

        # 3. Merge with Precautions
        if os.path.exists('symptom_precaution.csv'):
            df_pre = pd.read_csv('symptom_precaution.csv')
            df_pre.rename(columns={'Disease': 'disease_name'}, inplace=True)
            df_pre['disease_name'] = df_pre['disease_name'].str.strip()
            # Combine all 4 precaution columns into one
            pre_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
            df_pre['precautions'] = df_pre[pre_cols].apply(
                lambda x: ', '.join([str(val).strip() for val in x if pd.notna(val) and str(val).lower() != 'nan']), axis=1
            )
            df = pd.merge(df, df_pre[['disease_name', 'precautions']], on='disease_name', how='left')

        # 4. Add Mock Tablets and Hinglish (As these aren't in the Kaggle files)
        # We can also add a simple logic to generate Hinglish messages
        df['tablets'] = "Consult a doctor for specific dosage"
        df['message_hinglish'] = df['disease_name'].apply(lambda x: f"Aapke lakshano se lagta hai ki aapko {x} ho sakta hai.")

        # Save the processed dataset for reference and debugging
        df.to_csv('processed_dataset.csv', index=False)
        logger.info(f"Successfully merged data. Total records: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None

def train_and_export_model():
    """Trains the NLP classifier using the merged dataset."""
    df = clean_and_merge_data()

    if df is None or df.empty:
        logger.error("No data available for training. Make sure dataset.csv exists.")
        return

    X = df['symptoms']
    y = df['disease_name']

    logger.info("Initializing NLP pipeline (TF-IDF + Random Forest)...")
    # n_jobs=-1 for faster training using all cores
    model = make_pipeline(
        TfidfVectorizer(stop_words='english'), 
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )

    logger.info("Training model on the full medical dataset. This might take a few seconds...")
    model.fit(X, y)

    model_path = 'disease_model.pkl'
    joblib.dump(model, model_path)
    logger.info(f"Model training complete. Serialized model saved to '{model_path}'.")

if __name__ == "__main__":
    train_and_export_model()

