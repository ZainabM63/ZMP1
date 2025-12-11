import os
import pandas as pd
import numpy as np
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Filter warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# CONFIG
OUTPUT_MODEL = os.path.join("model", "phisher_model.h5")
OUTPUT_TFLITE = os.path.join("model", "phisher_model.tflite")
OUTPUT_VECTORIZER = os.path.join("model", "phisher_vectorizer.pkl")
OUTPUT_ENCODER = os.path.join("model", "label_encoder.pkl")
RANDOM_STATE = 42

# Import shared utilities
from utils import extract_phishing_features, clean_text


def get_memory_usage():
    """Get current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024 / 1024  # Convert to GB


def load_all_datasets(sample_size=None):
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
                'malware': 'phishing',
                'defacement': 'phishing'
            }
            
            df_url['label'] = df_url['type'].str.lower().str.strip().map(type_mapping)
            df_url = df_url[df_url['label'].notna()]
            
            print(f"   ✓ url_dataset.csv: {len(df_url)} rows (after filtering)")
            
            # Sample URL dataset if too large
            if sample_size and len(df_url) > sample_size:
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
    
    # 4. Load SMSSmishCollection.txt (SMS/Smishing dataset)
    try:
        smish_path = "data/SMSSmishCollection.txt"
        if os.path.exists(smish_path):
            with open(smish_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            print(f"   ✓ SMSSmishCollection.txt: {len(lines)} rows (raw)")
            
            smish_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        label_text, message = parts
                        label_lower = label_text.lower().strip()
                        if 'smish' in label_lower or 'phish' in label_lower:
                            label = 'phishing'
                        elif 'spam' in label_lower:
                            label = 'spam'
                        elif 'ham' in label_lower or 'legit' in label_lower:
                            label = 'legitimate'
                        else:
                            label = 'phishing'
                        
                        all_data.append({
                            'text': message,
                            'label': label,
                            'original_text': message
                        })
                        smish_count += 1
                else:
                    all_data.append({
                        'text': line,
                        'label': 'phishing',
                        'original_text': line
                    })
                    smish_count += 1
            
            print(f"   ✓ SMSSmishCollection.txt: {smish_count} messages loaded")
        else:
            print(f"   ℹ️  SMSSmishCollection.txt not found (optional)")
    except Exception as e:
        print(f"   ✗ Error loading SMSSmishCollection.txt: {e}")
    
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
    
    # TF-IDF features with reduced max_features for memory efficiency
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
    combined_features = hstack([tfidf_features, phishing_df.values])
    
    print(f"   ✓ TF-IDF features: {tfidf_features.shape[1]}")
    print(f"   ✓ Phishing features: {phishing_df.shape[1]}")
    print(f"   ✓ Total features: {combined_features.shape[1]}")
    
    return combined_features, vectorizer, phishing_df.columns.tolist()


def build_model(input_dim, num_classes=3):
    """Build TensorFlow/Keras model"""
    print("\n🤖 Building TensorFlow model...")
    
    model = Sequential([
        # Input layer
        Input(shape=(input_dim,), name='input'),
        
        # Hidden layers with dropout for regularization
        Dense(512, activation='relu', name='dense_1'),
        BatchNormalization(),
        Dropout(0.3, name='dropout_1'),
        
        Dense(256, activation='relu', name='dense_2'),
        BatchNormalization(),
        Dropout(0.3, name='dropout_2'),
        
        Dense(128, activation='relu', name='dense_3'),
        BatchNormalization(),
        Dropout(0.2, name='dropout_3'),
        
        # Output layer
        Dense(num_classes, activation='softmax', name='output')
    ], name='phishing_detector')
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   ✓ Model built with input dimension: {input_dim}")
    print(f"   ✓ Number of classes: {num_classes}")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
    """Train TensorFlow model with SMOTE balancing and sampling"""
    print("\n🎓 Training model...")
    
    # Apply SMOTE
    print("⚖️  Applying SMOTE to balance classes...")
    print("   Before SMOTE:")
    for i, count in enumerate(np.bincount(y_train)):
        print(f"      Class {i}: {count:,}")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print("   After SMOTE:")
    for i, count in enumerate(np.bincount(y_train_balanced)):
        pct = count / len(y_train_balanced) * 100
        print(f"      Class {i}: {count:,} ({pct:.1f}%)")
    
    # Sample to reduce memory usage with stratification
    MAX_TRAIN_SAMPLES = 60000
    if X_train_balanced.shape[0] > MAX_TRAIN_SAMPLES:
        print(f"\n💾 Sampling {MAX_TRAIN_SAMPLES:,} samples for memory efficiency...")
        # Use stratified sampling to maintain class balance
        from sklearn.model_selection import train_test_split
        X_train_balanced, _, y_train_balanced, _ = train_test_split(
            X_train_balanced, y_train_balanced,
            train_size=MAX_TRAIN_SAMPLES,
            stratify=y_train_balanced,
            random_state=42
        )
        print(f"   ✓ Training set reduced to {X_train_balanced.shape[0]:,} samples")
        print("   After sampling:")
        for i, count in enumerate(np.bincount(y_train_balanced)):
            pct = count / len(y_train_balanced) * 100
            print(f"      Class {i}: {count:,} ({pct:.1f}%)")
    
    # Convert sparse matrices to dense arrays
    print(f"\n🔄 Converting sparse matrices to dense arrays...")
    print(f"   Current memory usage: {get_memory_usage():.2f} GB")
    
    X_train_dense = X_train_balanced.toarray().astype(np.float32)
    X_test_dense = X_test.toarray().astype(np.float32)
    
    print(f"   After conversion: {get_memory_usage():.2f} GB")
    print(f"   ✓ Training set shape: {X_train_dense.shape}")
    print(f"   ✓ Test set shape: {X_test_dense.shape}")
    
    # Train model
    print(f"\n🎯 Training for {epochs} epochs...")
    history = model.fit(
        X_train_dense, y_train_balanced,
        validation_data=(X_test_dense, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return model, history


def evaluate_model(model, X_test, y_test, label_encoder, output_dir="model"):
    """Comprehensive model evaluation"""
    print("\n📊 Evaluating model...")
    
    # Convert test data to dense if needed
    if hasattr(X_test, 'toarray'):
        X_test_dense = X_test.toarray().astype(np.float32)
    else:
        X_test_dense = X_test.astype(np.float32)
    
    # Predictions
    y_pred_proba = model.predict(X_test_dense, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print("\n📋 Per-class metrics:")
    classes = label_encoder.classes_
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes))
    )
    
    for i, class_name in enumerate(classes):
        print(f"  {class_name:12s} - Precision: {precision[i]:.4f}  Recall: {recall[i]:.4f}  F1: {f1[i]:.4f}  Support: {support[i]}")
    
    # Classification report
    print("\n" + classification_report(y_test, y_pred, target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes, output_dir)
    
    # Sample predictions
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
    
    # Create annotations
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


def convert_to_tflite(model, tflite_path):
    """Convert Keras model to TFLite format"""
    print("\n🔄 Converting model to TFLite...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"   ✓ TFLite model saved to {tflite_path}")
        print(f"   ✓ Model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
        return True
    except Exception as e:
        print(f"   ✗ Error converting to TFLite: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tflite_model(tflite_path, X_test, y_test, label_encoder):
    """Test the TFLite model to ensure it works"""
    print("\n🧪 Testing TFLite model...")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"   ✓ TFLite model loaded successfully")
        print(f"   ✓ Input shape: {input_details[0]['shape']}")
        print(f"   ✓ Output shape: {output_details[0]['shape']}")
        
        # Test with a few samples
        test_samples = min(10, X_test.shape[0])
        correct = 0
        
        print(f"\n   Testing {test_samples} samples:")
        for i in range(test_samples):
            # Prepare input - convert sparse to dense for this sample only
            if hasattr(X_test, 'toarray'):
                input_data = X_test[i:i+1].toarray().astype(np.float32)
            else:
                input_data = X_test[i:i+1].astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_class = np.argmax(output_data[0])
            true_class = y_test[i]
            confidence = np.max(output_data[0]) * 100
            
            match = "✓" if pred_class == true_class else "✗"
            true_label = label_encoder.inverse_transform([true_class])[0]
            pred_label = label_encoder.inverse_transform([pred_class])[0]
            
            print(f"      {match} True: {true_label:12s} | Pred: {pred_label:12s} | Confidence: {confidence:.1f}%")
            
            if pred_class == true_class:
                correct += 1
        
        accuracy = correct / test_samples * 100
        print(f"\n   ✓ TFLite test accuracy: {correct}/{test_samples} ({accuracy:.1f}%)")
        
        if accuracy >= 80:
            print("   ✓ TFLite model is working correctly!")
        else:
            print("   ⚠️  TFLite accuracy is lower than expected")
        
        return accuracy
        
    except Exception as e:
        print(f"   ✗ Error testing TFLite model: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    os.makedirs("model", exist_ok=True)
    
    print("="*80)
    print("🚀 Phishing Detection Model Training (TensorFlow)")
    print("="*80)
    
    # Load data
    df = load_all_datasets(sample_size=80000)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])
    
    # Create features
    X, vectorizer, feature_names = create_feature_matrix(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Build model
    num_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]
    model = build_model(input_dim, num_classes)
    
    # Train model (SMOTE and sampling happen inside)
    model, history = train_model(model, X_train, y_train, X_test, y_test, epochs=10)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, label_encoder)
    
    # Save model
    print("\n💾 Saving model...")
    model.save(OUTPUT_MODEL)
    print(f"   ✓ Model saved to {OUTPUT_MODEL}")
    
    with open(OUTPUT_VECTORIZER, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"   ✓ Vectorizer saved to {OUTPUT_VECTORIZER}")
    
    with open(OUTPUT_ENCODER, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   ✓ Label encoder saved to {OUTPUT_ENCODER}")
    
    # Convert to TFLite
    if convert_to_tflite(model, OUTPUT_TFLITE):
        # Test TFLite model
        test_tflite_model(OUTPUT_TFLITE, X_test, y_test, label_encoder)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()
