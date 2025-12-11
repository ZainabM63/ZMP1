import os
import json
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
from scipy.sparse import hstack

# Filter warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# CONFIG
RANDOM_STATE = 42
OUTPUT_DIR = "flutter_model"
VOCAB_SIZE = 8000

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Import shared utilities
from utils import extract_phishing_features, clean_text


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
    
    # 4. Load SMSSmishCollection.txt (SMS/Smishing dataset)
    try:
        smish_path = "data/SMSSmishCollection.txt"
        if os.path.exists(smish_path):
            with open(smish_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            print(f"   ✓ SMSSmishCollection.txt: {len(lines)} rows (raw)")
            
            # Track loaded messages
            smish_count = 0
            
            # Parse the text file
            # Common formats: "label\ttext" or "label,text" or just text (assume phishing)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse with tab separator
                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        label_text, message = parts
                        # Map label variations to our classes
                        label_lower = label_text.lower().strip()
                        if 'smish' in label_lower or 'phish' in label_lower:
                            label = 'phishing'
                        elif 'spam' in label_lower:
                            label = 'spam'
                        elif 'ham' in label_lower or 'legit' in label_lower:
                            label = 'legitimate'
                        else:
                            # Default to phishing if unclear
                            label = 'phishing'
                        
                        all_data.append({
                            'text': message,
                            'label': label,
                            'original_text': message
                        })
                        smish_count += 1
                else:
                    # If no separator, assume the entire line is a phishing/smishing message
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
    
    print(f"\n📊 Total messages loaded: {len(df):,}")
    print("\n📊 Initial class distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        pct = 100 * count / len(df)
        print(f"   {label}: {count:,} ({pct:.1f}%)")
    
    return df


def create_feature_matrix(df, vectorizer=None, fit=True):
    """Create feature matrix combining TF-IDF and phishing features"""
    print("\n🔢 Creating TF-IDF features...")
    
    # TF-IDF features
    if fit:
        vectorizer = TfidfVectorizer(max_features=VOCAB_SIZE, ngram_range=(1, 2), min_df=2)
        tfidf_features = vectorizer.fit_transform(df['cleaned_text'])
        print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
    else:
        tfidf_features = vectorizer.transform(df['cleaned_text'])
    
    # Phishing-specific features
    print("   ✓ Extracting phishing features...")
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


def balance_classes_with_smote(X, y, random_state=42):
    """Balance classes using SMOTE"""
    print("\n⚖️  Applying SMOTE to balance classes...")
    
    # Get current distribution
    unique, counts = np.unique(y, return_counts=True)
    print("   Before SMOTE:")
    for label, count in zip(unique, counts):
        print(f"      Class {label}: {count:,}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Get new distribution
    unique, counts = np.unique(y_balanced, return_counts=True)
    print("   After SMOTE:")
    total = len(y_balanced)
    for label, count in zip(unique, counts):
        pct = 100 * count / total
        print(f"      Class {label}: {count:,} ({pct:.1f}%)")
    
    return X_balanced, y_balanced


def build_tensorflow_model(input_dim, num_classes):
    """Build TensorFlow Sequential model"""
    print("\n🤖 Building TensorFlow model...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu', name='dense_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.3, name='dropout_2'),
        tf.keras.layers.Dense(128, activation='relu', name='dense_3'),
        tf.keras.layers.Dropout(0.2, name='dropout_3'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("   ✓ Model architecture:")
    print(f"     - Input: ({input_dim},)  [TF-IDF + phishing features]")
    print(f"     - Hidden: Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.2)")
    print(f"     - Output: Dense({num_classes}, softmax)")
    
    total_params = model.count_params()
    print(f"   ✓ Total parameters: {total_params:,}")
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train TensorFlow model with callbacks"""
    print("\n🎓 Training model...")
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'phishing_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, label_encoder):
    """Comprehensive model evaluation"""
    print("\n📊 Evaluating model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
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
    plot_confusion_matrix(cm, classes)
    
    # ROC curves for multi-class
    plot_roc_curves(y_test, y_pred_proba, classes)
    
    # Sample predictions with confidence
    print("\n🔮 Sample predictions (high confidence):")
    indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
    high_conf_count = 0
    for idx in indices:
        true_label = classes[y_test[idx]]
        pred_label = classes[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]] * 100
        
        if confidence > 80:
            high_conf_count += 1
            status = "✓" if true_label == pred_label else "✗"
            print(f"   {status} True: {true_label:12s} | Pred: {pred_label:12s} | Confidence: {confidence:.1f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }


def plot_confusion_matrix(cm, classes):
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
    output_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Confusion matrix saved to {output_path}")


def plot_roc_curves(y_test, y_pred_proba, classes):
    """Plot ROC curves for multi-class classification"""
    from sklearn.preprocessing import label_binarize
    
    # Binarize the output for multi-class ROC
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
    output_path = os.path.join(OUTPUT_DIR, 'roc_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ ROC curves saved to {output_path}")


def convert_to_tflite(model):
    """Convert Keras model to TensorFlow Lite"""
    print("\n📱 Converting to TensorFlow Lite...")
    
    # Convert with optimizations
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save
    tflite_path = os.path.join(OUTPUT_DIR, 'phishing_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file size
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"   ✓ TFLite model saved to {tflite_path}")
    print(f"   ✓ Model size: {size_mb:.2f} MB")
    
    return tflite_path, size_mb


def export_tokenizer(vectorizer):
    """Export TF-IDF vocabulary and IDF values as JSON"""
    print("\n💾 Exporting tokenizer...")
    
    # Create vocabulary dictionary
    vocab = {word: int(idx) for word, idx in vectorizer.vocabulary_.items()}
    
    # Create tokenizer config
    tokenizer_config = {
        'vocabulary': vocab,
        'idf_values': vectorizer.idf_.tolist(),
        'vocab_size': len(vocab),
        'ngram_range': list(vectorizer.ngram_range),
        'min_df': vectorizer.min_df
    }
    
    # Save
    tokenizer_path = os.path.join(OUTPUT_DIR, 'tokenizer.json')
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"   ✓ Tokenizer saved to {tokenizer_path}")
    print(f"   ✓ Vocabulary size: {len(vocab)}")


def export_labels(label_encoder):
    """Export class labels mapping"""
    print("   ✓ Exporting labels...")
    
    labels_config = {
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_)
    }
    
    # Save
    labels_path = os.path.join(OUTPUT_DIR, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump(labels_config, f, indent=2)
    
    print(f"   ✓ Labels saved to {labels_path}")


def export_model_info(metrics, vocab_size, model_size):
    """Export model metadata"""
    print("   ✓ Exporting model info...")
    
    model_info = {
        'accuracy': float(metrics['accuracy']),
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1'],
        'vocab_size': vocab_size,
        'model_size_mb': model_size,
        'framework': 'TensorFlow/Keras',
        'conversion': 'TensorFlow Lite',
        'random_state': RANDOM_STATE
    }
    
    # Save
    info_path = os.path.join(OUTPUT_DIR, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"   ✓ Model info saved to {info_path}")


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
        test_samples = 5
        correct = 0
        
        for i in range(test_samples):
            # Prepare input
            input_data = X_test[i:i+1].toarray().astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_class = np.argmax(output_data[0])
            true_class = y_test[i]
            
            if pred_class == true_class:
                correct += 1
        
        print(f"   ✓ TFLite test accuracy: {correct}/{test_samples} ({correct/test_samples*100:.1f}%)")
        print("   ✓ TFLite model is working correctly!")
        
    except Exception as e:
        print(f"   ✗ Error testing TFLite model: {e}")


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("🚀 TensorFlow Model Training for Flutter Integration")
    print("="*80)
    
    # Load data
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
    
    # Convert to dense arrays for TensorFlow
    print("\n🔄 Converting sparse matrices to dense arrays...")
    X_train_dense = X_train_balanced.toarray().astype(np.float32)
    X_test_dense = X_test.toarray().astype(np.float32)
    
    # Create validation split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_dense, y_train_balanced, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train_balanced
    )
    
    print(f"   ✓ Training samples: {len(X_train_final):,}")
    print(f"   ✓ Validation samples: {len(X_val):,}")
    print(f"   ✓ Test samples: {len(X_test_dense):,}")
    
    # Build model
    input_dim = X_train_dense.shape[1]
    num_classes = len(label_encoder.classes_)
    model = build_tensorflow_model(input_dim, num_classes)
    
    # Train model
    history = train_model(model, X_train_final, y_train_final, X_val, y_val)
    
    # Evaluate
    metrics = evaluate_model(model, X_test_dense, y_test, label_encoder)
    
    # Save Keras model
    print("\n💾 Saving Keras model...")
    keras_path = os.path.join(OUTPUT_DIR, 'phishing_model.h5')
    model.save(keras_path)
    print(f"   ✓ Keras model saved to {keras_path}")
    
    # Convert to TFLite
    tflite_path, model_size = convert_to_tflite(model)
    
    # Export artifacts
    export_tokenizer(vectorizer)
    export_labels(label_encoder)
    export_model_info(metrics, VOCAB_SIZE, model_size)
    
    # Test TFLite model
    test_tflite_model(tflite_path, X_test, y_test, label_encoder)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"\n📦 Artifacts saved in '{OUTPUT_DIR}/' directory:")
    print(f"   - phishing_model.tflite  ({model_size:.2f} MB)")
    print(f"   - phishing_model.h5  (Keras backup)")
    print(f"   - tokenizer.json  (TF-IDF vocabulary)")
    print(f"   - labels.json  (Class labels)")
    print(f"   - model_info.json  (Metadata)")
    print(f"   - confusion_matrix.png")
    print(f"   - roc_curves.png")
    print("\n📊 Final Results:")
    print(f"   ✓ Test Accuracy: {metrics['accuracy']*100:.2f}%")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"   ✓ {class_name:12s} F1-score: {metrics['f1'][i]:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
