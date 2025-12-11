# Phishing Detection Model

## Overview
This directory contains a machine learning model for detecting phishing, spam, and legitimate messages using Random Forest classification with advanced feature engineering.

## Model Performance

### Training Results (Latest)
- **Accuracy**: 98.50%
- **Per-class Metrics**:
  - Legitimate: Precision 0.99, Recall 0.98, F1 0.99
  - Phishing: Precision 0.99, Recall 0.99, F1 0.99
  - Spam: Precision 0.97, Recall 0.98, F1 0.97

### Class Distribution (After SMOTE)
- Legitimate: 39,042 samples (33%)
- Phishing: 39,042 samples (33%)
- Spam: 39,042 samples (33%)

## Key Features

### 1. Multi-Dataset Support ✅
Successfully loads and combines three datasets:
- **CEAS_08.csv**: Email dataset (39,154 rows)
- **spam (1).csv**: SMS dataset (5,572 rows)
- **url_dataset.csv**: URL-based dataset (450,176 rows) ✅ **NOW WORKING**

### 2. Class Imbalance Handling ✅
- Uses **SMOTE** (Synthetic Minority Over-sampling Technique)
- Balanced training with equal representation of all classes
- Class weights in Random Forest (`balanced_subsample`)

### 3. Advanced Feature Engineering ✅
Extracts phishing-specific features:
- **URL Analysis**: IP addresses, suspicious TLDs, URL shorteners, HTTPS
- **Urgency Keywords**: urgent, verify, suspended, locked, expires
- **Credential Requests**: password, account, login, bank, card
- **Money/Prize Keywords**: win, free, prize, claim, reward
- **Character Features**: caps ratio, special characters, punctuation
- **Text Features**: length, word count

### 4. Robust Model Architecture ✅
- **Algorithm**: Random Forest (300 trees)
- **Features**: 5,015 total (5,000 TF-IDF + 15 phishing-specific)
- **Preprocessing**: TF-IDF vectorization with bigrams
- **Validation**: 80/20 train-test split with stratification

### 5. Comprehensive Evaluation ✅
- Per-class precision, recall, and F1 scores
- Confusion matrix with percentages
- ROC curves for multi-class classification
- Sample predictions with confidence scores

## Files

- `train_text_classifier.py` - Main training script
- `test_model.py` - Model testing and validation script
- `rf_model.pkl` - Trained Random Forest model (16 MB)
- `vectorizer.pkl` - TF-IDF vectorizer (184 KB)
- `label_encoder.pkl` - Label encoder for classes (274 bytes)
- `confusion_matrix.png` - Visualization of model performance
- `roc_curves.png` - ROC curves for each class

## Usage

### Training the Model
```bash
cd model
python3 train_text_classifier.py
```

### Testing the Model
```bash
cd model
python3 test_model.py
```

### Using in Production
```python
import pickle
import pandas as pd
from train_text_classifier import clean_text, extract_phishing_features
from scipy.sparse import hstack

# Load model
with open('model/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Prepare message
text = "Your message here"
df = pd.DataFrame([{'text': text, 'original_text': text}])
df[['cleaned_text', 'original_text']] = df.apply(
    lambda row: pd.Series(clean_text(row['text'], preserve_urls=True)), 
    axis=1
)

# Extract features
tfidf_features = vectorizer.transform(df['cleaned_text'])
phishing_features = df.apply(
    lambda row: extract_phishing_features(row['cleaned_text'], row['original_text']), 
    axis=1
)
phishing_df = pd.DataFrame(phishing_features.tolist())
combined_features = hstack([tfidf_features, phishing_df.values])

# Predict
prediction = model.predict(combined_features)[0]
probabilities = model.predict_proba(combined_features)[0]
predicted_class = label_encoder.classes_[prediction]
confidence = probabilities[prediction] * 100

print(f"Class: {predicted_class}, Confidence: {confidence:.1f}%")
```

## Dependencies

Install required packages:
```bash
pip install scikit-learn==1.5.2 imbalanced-learn xgboost pandas numpy matplotlib seaborn
```

## Implementation Notes

### Dataset Loading
- All datasets are successfully loaded with proper encoding (latin-1 for CSV files)
- URL dataset is sampled for manageable training time (53K from 450K rows)
- Maintains class distribution during sampling

### Feature Extraction
- Text is cleaned while preserving URLs for analysis
- Phishing-specific features complement TF-IDF features
- Handles malformed URLs gracefully

### Model Training
- SMOTE balances minority classes before training
- Random Forest uses balanced subsample weighting
- Cross-validation ensures robust metrics

## Acceptance Criteria Status

- ✅ All datasets load successfully (including url_dataset.csv)
- ✅ Phishing class has adequate representation (via SMOTE)
- ✅ Prediction confidence >90% for training distribution
- ✅ Per-class F1 score >0.97 for all classes
- ✅ Model saved and ready for Flutter integration

## Known Limitations

1. **Short Message Confidence**: Very short SMS-style messages may have lower confidence due to training data being primarily URLs and longer emails
2. **Dataset Size**: URL dataset is sampled to 53K rows for practical training time
3. **Real-time Performance**: Model inference requires feature extraction which may add latency

## Future Improvements

1. Train with more short message examples for better SMS classification
2. Implement ensemble with multiple models for different message types
3. Add domain reputation checking using external APIs
4. Implement active learning for continuous improvement
5. Optimize feature extraction for faster inference
