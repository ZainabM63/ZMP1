
import os
import sys
import re
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "phisher_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "phisher_vectorizer. pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# Safe CSV field size limit for Windows
try:
    import csv
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
except: 
    pass

def load_csv_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV dataset.  Handles common formats: 
    - v1, v2 (SMS spam format)
    - label, text
    - message, label
    """
    try:
        # Try reading with latin-1 encoding (common for SMS datasets)
        df = pd.read_csv(csv_path, encoding='latin-1', on_bad_lines='skip')
    except: 
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Find label and text columns
    label_col = None
    text_col = None
    
    for col in df.columns:
        if col in ['v1', 'label', 'class', 'category', 'type']:
            label_col = col
        if col in ['v2', 'text', 'message', 'content', 'sms', 'email', 'body']:
            text_col = col
    
    if not label_col or not text_col:
        # Fallback:  assume first two columns
        if len(df.columns) >= 2:
            label_col = df.columns[0]
            text_col = df.columns[1]
        else:
            raise ValueError(f"Cannot identify label/text columns in {csv_path. name}")
    
    # Extract only needed columns
    df = df[[label_col, text_col]].copy()
    df. columns = ['label', 'text']
    df = df.dropna()
    
    # Normalize labels
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    df['text'] = df['text']. astype(str).str.strip()
    
    # Map to standard labels
    label_map = {
        'ham': 'legitimate',
        'legit': 'legitimate',
        'legitimate': 'legitimate',
        'normal': 'legitimate',
        'safe': 'legitimate',
        '0': 'legitimate',
        
        'spam': 'spam',
        'junk': 'spam',
        '1': 'spam',
        
        'phishing': 'phishing',
        'phish': 'phishing',
        'smishing': 'phishing',
        'malicious': 'phishing',
        'fraud': 'phishing',
        'scam': 'phishing',
        '2': 'phishing',
    }
    
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    
    df['source'] = csv_path.name
    return df[['text', 'label', 'source']]


def load_txt_dataset(txt_path: Path) -> pd.DataFrame:
    
    messages = []
    
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try tab-separated
            if '\t' in line: 
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    label, text = parts
                    label = label.lower().strip()
                    
                    # Map to standard label
                    if any(x in label for x in ['phish', 'smish', 'malicious', 'fraud', 'scam']):
                        label = 'phishing'
                    elif any(x in label for x in ['ham', 'legit', 'safe', 'normal']):
                        label = 'legitimate'
                    elif 'spam' in label:
                        label = 'spam'
                    else:
                        label = 'phishing'  # Default for smishing datasets
                    
                    messages.append({'text': text.strip(), 'label': label})
                    continue
            
            # Try comma-separated
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    label, text = parts
                    label = label.lower().strip()
                    
                    if any(x in label for x in ['phish', 'smish', 'malicious', 'fraud', 'scam']):
                        label = 'phishing'
                    elif any(x in label for x in ['ham', 'legit', 'safe', 'normal']):
                        label = 'legitimate'
                    elif 'spam' in label:
                        label = 'spam'
                    else:
                        label = 'phishing'
                    
                    messages.append({'text': text.strip(), 'label': label})
                    continue
            
            # Plain text - assume phishing (for smishing datasets)
            messages.append({'text': line, 'label': 'phishing'})
    
    if not messages:
        raise ValueError(f"No valid messages found in {txt_path. name}")
    
    df = pd.DataFrame(messages)
    df['source'] = txt_path.name
    return df[['text', 'label', 'source']]


def load_all_datasets() -> pd.DataFrame:
    """Load all CSV and TXT datasets from data directory"""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found.  Please create it and add datasets.")
    
    datasets = []
    
    # Load CSV files
    csv_files = list(DATA_DIR.glob('*.csv'))
    for csv_file in csv_files:
        print(f"📂 Loading CSV: {csv_file.name}")
        try:
            df = load_csv_dataset(csv_file)
            print(f"   ✓ {len(df)} messages | Classes: {df['label'].value_counts().to_dict()}")
            datasets. append(df)
        except Exception as e:
            print(f"   ✗ Error:  {e}")
    
    # Load TXT files
    txt_files = list(DATA_DIR. glob('*.txt'))
    for txt_file in txt_files: 
        print(f"📂 Loading TXT: {txt_file.name}")
        try:
            df = load_txt_dataset(txt_file)
            print(f"   ✓ {len(df)} messages | Classes: {df['label'].value_counts().to_dict()}")
            datasets.append(df)
        except Exception as e: 
            print(f"   ✗ Error: {e}")
    
    if not datasets:
        raise ValueError("No datasets loaded. Add CSV or TXT files to the 'data/' directory.")
    
    # Combine all datasets
    df_combined = pd.concat(datasets, ignore_index=True)
    
    print(f"\n📊 Combined:  {len(df_combined)} total messages")
    print("\nClass distribution:")
    print(df_combined['label'].value_counts())
    
    return df_combined


def preprocess_text(text:  str) -> str:
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace URLs with token
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' url_token ', text)
    
    # Replace emails with token
    text = re. sub(r'\S+@\S+', ' email_token ', text)
    
    # Replace phone numbers with token
    text = re.sub(r'\b\d{10,}\b', ' phone_token ', text)
    
    # Replace numbers with token
    text = re. sub(r'\d+', ' num ', text)
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Collapse multiple spaces
    text = re. sub(r'\s+', ' ', text).strip()
    
    return text


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional features from text"""
    df = df.copy()
    
    # Length features
    df['char_count'] = df['text']. str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    # URL features
    df['has_url'] = df['text']. str.contains(r'http|www', case=False, na=False).astype(int)
    df['url_count'] = df['text'].str.count(r'http[s]?://\S+|www\.\S+')
    
    # Urgency keywords
    urgency_words = ['urgent', 'immediate', 'act now', 'limited time', 'expires', 'winner', 'congratulations', 'claim', 'verify', 'suspend', 'locked']
    df['has_urgency'] = df['text'].str.lower().str.contains('|'.join(urgency_words), na=False).astype(int)
    
    # Money/prize keywords
    money_words = ['prize', 'win', 'won', 'free', 'cash', '£', '$', '€', 'thousand', 'million', 'reward']
    df['has_money'] = df['text'].str.lower().str.contains('|'.join(money_words), na=False).astype(int)
    
    # Credential keywords
    credential_words = ['password', 'account', 'verify', 'confirm', 'login', 'security', 'bank', 'card', 'paypal']
    df['has_credential'] = df['text'].str.lower().str.contains('|'.join(credential_words), na=False).astype(int)
    
    return df


def train_model():
    """Complete training pipeline"""
    print("="*70)
    print(" THE PHISHER - Training Multi-Class Detector ". center(70))
    print("="*70)
    print()
    
    # Create model directory
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load datasets
    df = load_all_datasets()
    
    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"\n🧹 Removed {before - len(df)} duplicates → {len(df)} unique messages\n")
    
    # Preprocess
    print("🔄 Preprocessing text...")
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    # Extract features (optional, for reference)
    df = extract_features(df)
    
    # Prepare train/test split
    X = df['text_clean']
    y = df['label']
    
    # Check class distribution
    class_counts = y.value_counts()
    print(f"\n📊 Final class distribution:")
    for label, count in class_counts.items():
        print(f"   {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✂️  Split: {len(X_train)} train | {len(X_test)} test")
    
    # Vectorize
    print("\n🔢 Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents='unicode'
    )
    
    X_train_vec = vectorizer. fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train model
    print("\n🤖 Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    clf.fit(X_train_vec, y_train)
    
    # Evaluate
    print("\n" + "="*70)
    print(" EVALUATION RESULTS ".center(70))
    print("="*70)
    
    y_pred = clf.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['legitimate', 'spam', 'phishing'])
    print(cm)
    
    # Save model artifacts
    print("\n💾 Saving model artifacts...")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"   ✓ Model → {MODEL_PATH}")
    
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"   ✓ Vectorizer → {VECTORIZER_PATH}")
    
    # Save label encoder (classes)
    label_encoder = {'classes': clf.classes_. tolist()}
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   ✓ Label encoder → {LABEL_ENCODER_PATH}")
    
    # Test samples
    print("\n" + "="*70)
    print(" SAMPLE PREDICTIONS ".center(70))
    print("="*70)
    
    test_samples = [
        "Hey, are you coming to dinner tonight?",
        "WINNER! You've won £1000 cash prize.  Call 09XX to claim now! ",
        "Your bank account has been suspended.  Verify immediately:  http://fake-bank.com",
        "Meeting rescheduled to 3pm. See you in room B.",
        "URGENT: Your package delivery failed. Track here: bit.ly/xyz123"
    ]
    
    for msg in test_samples:
        result = predict_single(msg, clf, vectorizer)
        print(f"\n📱 Message: {msg[: 65]}...")
        print(f"   🎯 Prediction: {result['prediction']. upper()}")
        print(f"   📊 Confidence:  {result['confidence']:.2%}")
        print(f"   📈 Probabilities: ", end="")
        for label, prob in result['probabilities']. items():
            print(f"{label}={prob:.2%}  ", end="")
        print()
    
    print("\n" + "="*70)
    print(" ✅ Training Complete!  ".center(70))
    print("="*70)



def load_trained_model() -> Tuple: 
    
    if not MODEL_PATH.exists() or not VECTORIZER_PATH. exists():
        raise FileNotFoundError(
            "Model files not found. Please train the model first:\n"
            "  python phisher_complete.py train"
        )
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def predict_single(message: str, model=None, vectorizer=None) -> Dict:
    """Predict single message"""
    # Load model if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_trained_model()
    
    # Preprocess
    text_clean = preprocess_text(message)
    
    # Vectorize
    text_vec = vectorizer.transform([text_clean])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    # Build result
    result = {
        'prediction': prediction,
        'confidence': float(max(probabilities)),
        'probabilities': {label: float(prob) for label, prob in zip(model.classes_, probabilities)}
    }
    
    return result


def predict_batch(messages: List[str]) -> List[Dict]:
    """Predict multiple messages"""
    model, vectorizer = load_trained_model()
    
    results = []
    for msg in messages:
        result = predict_single(msg, model, vectorizer)
        results.append(result)
    
    return results



def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train:    python phisher_complete.py train")
        print("  Predict: python phisher_complete.py predict <message>")
        print("  Example: python phisher_complete.py predict \"You won a prize! \"")
        sys.exit(1)
    
    command = sys.argv[1]. lower()
    
    if command == 'train':
        train_model()
    
    elif command == 'predict': 
        if len(sys.argv) < 3:
            print("Error: Please provide a message to predict")
            print("Example: python phisher_complete.py predict \"Your message here\"")
            sys.exit(1)
        
        message = ' '.join(sys.argv[2:])
        
        print("\n" + "="*70)
        print(" THE PHISHER - Prediction ". center(70))
        print("="*70)
        
        result = predict_single(message)
        
        print(f"\n📱 Message: {message}\n")
        print(f"🎯 Prediction: {result['prediction'].upper()}")
        print(f"📊 Confidence: {result['confidence']:.2%}\n")
        print("📈 All probabilities:")
        for label, prob in result['probabilities'].items():
            bar = '█' * int(prob * 40)
            print(f"   {label: 12s}: {bar} {prob:.2%}")
        
        print("\n" + "="*70)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict")
        sys.exit(1)


if __name__ == '__main__':
    main()