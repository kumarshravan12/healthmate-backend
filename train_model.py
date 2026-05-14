import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC          # 10x faster than RandomForest
from sklearn.calibration import CalibratedClassifierCV  # Adds predict_proba to SVC
from sklearn.pipeline import make_pipeline
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_and_merge_data():
    """Merges multiple CSV files into a single training dataset."""
    try:
        if not os.path.exists('dataset.csv'):
            logger.error("'dataset.csv' not found.")
            return None

        df_raw = pd.read_csv('dataset.csv')

        # Combine all symptom columns into a single text string
        symptom_cols = [c for c in df_raw.columns if 'Symptom' in c]
        df_raw['symptoms'] = df_raw[symptom_cols].apply(
            lambda x: ' '.join([
                str(val).replace('_', ' ').strip()
                for val in x if pd.notna(val) and str(val).lower() != 'nan'
            ]), axis=1
        )

        df = df_raw[['Disease', 'symptoms']].copy()
        df.rename(columns={'Disease': 'disease_name'}, inplace=True)
        df['disease_name'] = df['disease_name'].str.strip()

        # Merge descriptions
        if os.path.exists('symptom_Description.csv'):
            df_desc = pd.read_csv('symptom_Description.csv')
            df_desc.rename(columns={'Disease': 'disease_name'}, inplace=True)
            df_desc['disease_name'] = df_desc['disease_name'].str.strip()
            df = pd.merge(df, df_desc, on='disease_name', how='left')

        # Merge precautions
        if os.path.exists('symptom_precaution.csv'):
            df_pre = pd.read_csv('symptom_precaution.csv')
            df_pre.rename(columns={'Disease': 'disease_name'}, inplace=True)
            df_pre['disease_name'] = df_pre['disease_name'].str.strip()
            pre_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
            df_pre['precautions'] = df_pre[pre_cols].apply(
                lambda x: ', '.join([str(val).strip() for val in x if pd.notna(val) and str(val).lower() != 'nan']),
                axis=1
            )
            df = pd.merge(df, df_pre[['disease_name', 'precautions']], on='disease_name', how='left')

        df['tablets'] = "Consult a doctor for specific dosage"
        df['message_hinglish'] = df['disease_name'].apply(
            lambda x: f"Aapke lakshano se lagta hai ki aapko {x} ho sakta hai."
        )

        df.to_csv('processed_dataset.csv', index=False)
        logger.info(f"Data merged successfully. Total records: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None


def train_and_export_model():
    """Trains a fast LinearSVC NLP classifier."""
    df = clean_and_merge_data()

    if df is None or df.empty:
        logger.error("No data available for training.")
        return

    X = df['symptoms']
    y = df['disease_name']

    logger.info("Initializing fast NLP pipeline (TF-IDF + LinearSVC)...")

    # LinearSVC is 10x faster at prediction time than RandomForest.
    # CalibratedClassifierCV wraps it to enable predict_proba (confidence scores).
    model = make_pipeline(
        TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),   # bigrams are enough; trigrams slow it down
            max_features=8000,    # more features = better vocab coverage
            sublinear_tf=True     # apply log normalization for better accuracy
        ),
        CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0), cv=3)
    )

    logger.info("Training model... (this may take ~30 seconds)")
    model.fit(X, y)

    joblib.dump(model, 'disease_model.pkl')
    logger.info("Model saved to 'disease_model.pkl'. Training complete!")


if __name__ == "__main__":
    train_and_export_model()
