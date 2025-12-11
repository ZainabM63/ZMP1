import os
import re
import pandas as pd
import numpy as np
import pickle
import warnings
from urllib.parse import urlparse
from collections import Counter

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# CONFIG
OUTPUT_MODEL = os.path.join("model", "rf_model.pkl")
OUTPUT_VECTORIZER = os.path.join("model", "vectorizer.pkl")
OUTPUT_ENCODER = os.path.join("model", "label_encoder.pkl")
RANDOM_STATE = 42

# Phishing-specific keywords
URGENCY_KEYWORDS = ['urgent', 'verify', 'suspended', 'locked', 'expires', 'immediately', 'action required', 'limited time']
CREDENTIAL_KEYWORDS = ['password', 'account', 'login', 'bank', 'card', 'credit', 'ssn', 'social security', 'confirm']
MONEY_KEYWORDS = ['win', 'won', 'free', 'prize', 'claim', 'reward', 'bonus', 'lottery', 'million', 'inheritance']
SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work', '.click']

def extract_urls(text):
    """Extract URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def extract_phishing_features(text, original_text=None):
    """Extract phishing-specific features from text"""
    features = {}
    text_lower = text.lower() if text else ""
    original_lower = original_text.lower() if original_text else text_lower
    
    # URL features
    urls = extract_urls(original_lower)
    features['url_count'] = len(urls)
    features['has_url'] = int(len(urls) > 0)
    
    # Check for IP addresses in URLs
    features['has_ip_url'] = 0
    features['has_suspicious_tld'] = 0
    features['has_url_shortener'] = 0
    features['has_https'] = 0
    
    for url in urls:
        try:
            parsed = urlparse(url)
            # IP address check
            if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc):
                features['has_ip_url'] = 1
            # Suspicious TLD check
            if any(parsed.netloc.endswith(tld) for tld in SUSPICIOUS_TLDS):
                features['has_suspicious_tld'] = 1
            # URL shortener check
            if any(short in parsed.netloc for short in ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']):
                features['has_url_shortener'] = 1
            # HTTPS check
            if parsed.scheme == 'https':
                features['has_https'] = 1
        except (ValueError, Exception):
            # Skip malformed URLs
            continue
    
    # Keyword features
    features['urgency_count'] = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
    features['credential_count'] = sum(1 for kw in CREDENTIAL_KEYWORDS if kw in text_lower)
    features['money_count'] = sum(1 for kw in MONEY_KEYWORDS if kw in text_lower)
    
    # Character-level features
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['special_char_count'] = sum(1 for c in text if c in '!@#$%^&*')
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    
    # Length features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    return features

def clean_text(s: str, preserve_urls=False) -> tuple:
    """Clean text and optionally preserve URLs"""
    original = str(s)
    s = original.lower()
    
    if not preserve_urls:
        s = re.sub(r"http[s]?://\S+", " URL ", s)
    
    s = re.sub(r"[^a-z0-9\s@.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, original

def load_datasets(sample_size=None):
    """Load and combine all datasets with proper handling
    
    Args:
        sample_size: If provided, sample datasets to this size for faster training.
                    Use None for full dataset (default).
    """
    all_data = []
    
    print("📂 Loading datasets...")
    
    # 1. Load CEAS_08.csv (email dataset)
    try:
        df_ceas = pd.read_csv("data/CEAS_08.csv", encoding="latin-1")
        print(f"   ✓ CEAS_08.csv: {len(df_ceas)} rows")
        
        # Combine subject and body
        df_ceas['text'] = df_ceas['subject'].fillna('') + ' ' + df_ceas['body'].fillna('')
        df_ceas['label_class'] = df_ceas['label'].apply(lambda x: 'spam' if x == 1 else 'legitimate')
        
        for _, row in df_ceas.iterrows():
            all_data.append({
                'text': row['text'],
                'label': row['label_class'],
                'original_text': row['text']
            })
    except Exception as e:
        print(f"   ✗ Error loading CEAS_08.csv: {e}")
    
    # 2. Load spam (1).csv (SMS dataset)
    try:
        df_sms = pd.read_csv("data/spam (1).csv", encoding="latin-1")
        print(f"   ✓ spam (1).csv: {len(df_sms)} rows")
        
        df_sms = df_sms.rename(columns={"v1": "label", "v2": "text"})
        df_sms = df_sms[["label", "text"]].dropna()
        df_sms['label'] = df_sms['label'].str.lower().str.strip()
        df_sms['label'] = df_sms['label'].replace({'ham': 'legitimate'})
        
        for _, row in df_sms.iterrows():
            all_data.append({
                'text': row['text'],
                'label': row['label'],
                'original_text': row['text']
            })
    except Exception as e:
        print(f"   ✗ Error loading spam (1).csv: {e}")
    
    # 3. Load url_dataset.csv (URL-based dataset)
    try:
        df_url = pd.read_csv("data/url_dataset.csv", encoding="utf-8")
        print(f"   ✓ url_dataset.csv: {len(df_url)} rows (raw)")
        
        # Check columns
        if 'url' in df_url.columns and 'type' in df_url.columns:
            df_url = df_url.dropna(subset=['url', 'type'])
            
            # Map types to our classes
            type_mapping = {
                'legitimate': 'legitimate',
                'phishing': 'phishing',
                'spam': 'spam',
                'malware': 'phishing',  # treat malware as phishing
                'defacement': 'phishing'  # treat defacement as phishing
            }
            
            df_url['label'] = df_url['type'].str.lower().str.strip().map(type_mapping)
            df_url = df_url[df_url['label'].notna()]
            
            print(f"   ✓ url_dataset.csv: {len(df_url)} rows (after filtering)")
            
            # Sample URL dataset if too large (for performance)
            if sample_size and len(df_url) > sample_size:
                # Stratified sampling to maintain class distribution
                df_url = df_url.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // 3), random_state=RANDOM_STATE)
                )
                print(f"   ℹ️  Sampled url_dataset.csv to {len(df_url)} rows")
            
            # Use URL as text content
            for _, row in df_url.iterrows():
                all_data.append({
                    'text': row['url'],
                    'label': row['label'],
                    'original_text': row['url']
                })
        else:
            print(f"   ✗ url_dataset.csv missing expected columns. Found: {df_url.columns.tolist()}")
    except Exception as e:
        print(f"   ✗ Error loading url_dataset.csv: {e}")
    
    # Create final dataframe
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        raise ValueError("No data loaded from any dataset!")
    
    # Clean text
    print("\n🧹 Cleaning text...")
    df[['cleaned_text', 'original_text']] = df.apply(
        lambda row: pd.Series(clean_text(row['text'], preserve_urls=True)), 
        axis=1
    )
    
    print("\n📊 Initial class distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        pct = 100 * count / len(df)
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    return df

def create_feature_matrix(df, vectorizer=None, fit=True):
    """Create feature matrix combining TF-IDF and phishing features"""
    print("\n🔧 Extracting features...")
    
    # TF-IDF features
    if fit:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        tfidf_features = vectorizer.fit_transform(df['cleaned_text'])
    else:
        tfidf_features = vectorizer.transform(df['cleaned_text'])
    
    # Phishing-specific features
    phishing_features = df.apply(
        lambda row: extract_phishing_features(row['cleaned_text'], row['original_text']), 
        axis=1
    )
    phishing_df = pd.DataFrame(phishing_features.tolist())
    
    # Combine features
    from scipy.sparse import hstack
    combined_features = hstack([tfidf_features, phishing_df.values])
    
    print(f"   ✓ TF-IDF features: {tfidf_features.shape[1]}")
    print(f"   ✓ Phishing features: {phishing_df.shape[1]}")
    print(f"   ✓ Total features: {combined_features.shape[1]}")
    
    return combined_features, vectorizer, phishing_df.columns.tolist()

def balance_classes_with_smote(X, y, random_state=42):
    """Balance classes using SMOTE"""
    print("\n⚖️  Applying SMOTE to balance classes...")
    
    # Get current distribution
    unique, counts = np.unique(y, return_counts=True)
    print("   Before SMOTE:")
    for label, count in zip(unique, counts):
        print(f"      {label}: {count}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Get new distribution
    unique, counts = np.unique(y_balanced, return_counts=True)
    print("   After SMOTE:")
    for label, count in zip(unique, counts):
        print(f"      {label}: {count}")
    
    return X_balanced, y_balanced

def train_model(X_train, y_train):
    """Train ensemble model with balanced Random Forest"""
    print("\n🎓 Training model...")
    
    # Balanced Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=300,  # More trees for better performance
        max_depth=30,      # Deeper trees for complex patterns
        min_samples_split=5,  # Allow more granular splits
        min_samples_leaf=2,   # Smaller leaves for better detail
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    # Train
    rf.fit(X_train, y_train)
    print("   ✓ Random Forest trained")
    
    return rf

def evaluate_model(model, X_test, y_test, label_encoder, output_dir="model"):
    """Comprehensive model evaluation"""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print("\n📋 Per-class metrics:")
    classes = label_encoder.classes_
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=range(len(classes)))
    
    for i, class_name in enumerate(classes):
        print(f"  {class_name:12s} - Precision: {precision[i]:.4f}  Recall: {recall[i]:.4f}  F1: {f1[i]:.4f}  Support: {support[i]}")
    
    # Classification report
    print("\n" + classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes, output_dir)
    
    # ROC curves for multi-class
    plot_roc_curves(y_test, y_pred_proba, classes, output_dir)
    
    # Sample predictions with confidence
    print("\n🔮 Sample predictions (high confidence):")
    indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
    for idx in indices:
        true_label = classes[y_test[idx]]
        pred_label = classes[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]] * 100
        
        if confidence > 80:
            status = "✓" if true_label == pred_label else "✗"
            print(f"   {status} True: {true_label:12s} | Pred: {pred_label:12s} | Confidence: {confidence:.1f}%")

def plot_confusion_matrix(cm, classes, output_dir):
    """Plot confusion matrix with percentages"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Count and Percentage)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def plot_roc_curves(y_test, y_pred_proba, classes, output_dir):
    """Plot ROC curves for each class"""
    from sklearn.preprocessing import label_binarize
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(classes):
        if y_test_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ ROC curves saved to {os.path.join(output_dir, 'roc_curves.png')}")

def main():
    os.makedirs("model", exist_ok=True)
    
    print("="*80)
    print("🚀 Phishing Detection Model Training")
    print("="*80)
    
    # Load data with sampling for manageable training time
    # Use 80K total samples to ensure reasonable training time while maintaining quality
    # This gives us more text/SMS data (which are better for short message classification)
    df = load_datasets(sample_size=80000)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    
    # Create features
    X, vectorizer, feature_names = create_feature_matrix(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Balance training data with SMOTE
    X_train_balanced, y_train_balanced = balance_classes_with_smote(X_train, y_train, RANDOM_STATE)
    
    # Train model
    model = train_model(X_train_balanced, y_train_balanced)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save model
    print("\n💾 Saving model...")
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to {OUTPUT_MODEL}")
    
    with open(OUTPUT_VECTORIZER, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"   ✓ Vectorizer saved to {OUTPUT_VECTORIZER}")
    
    with open(OUTPUT_ENCODER, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   ✓ Label encoder saved to {OUTPUT_ENCODER}")
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80)

if __name__ == "__main__":
    main()