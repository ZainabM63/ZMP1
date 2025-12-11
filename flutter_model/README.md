# TensorFlow Lite Model for Flutter Integration

This directory contains a trained TensorFlow Lite model for phishing detection that can be integrated with Flutter mobile applications for 100% offline operation.

## Model Performance

- **Test Accuracy**: 99.19%
- **Model Size**: 4.09 MB (mobile-optimized)
- **Classes**: legitimate, phishing, spam

### Per-Class Metrics
| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| legitimate | 99.09%    | 99.41% | 99.25%   |
| phishing   | 99.44%    | 99.49% | 99.47%   |
| spam       | 99.11%    | 98.36% | 98.73%   |

## Files Description

### Model Files
- **`phishing_model.tflite`** (4.09 MB) - Optimized TensorFlow Lite model for mobile deployment
- **`phishing_model.h5`** (49 MB) - Original Keras model (backup/reference)

### Configuration Files
- **`tokenizer.json`** (349 KB) - TF-IDF vocabulary and IDF values for text preprocessing
- **`labels.json`** - Class label mapping (legitimate=0, phishing=1, spam=2)
- **`model_info.json`** - Model metadata including accuracy metrics

### Visualization Files
- **`confusion_matrix.png`** - Confusion matrix showing classification performance
- **`roc_curves.png`** - ROC curves with AUC scores (0.999-1.000)

## Model Architecture

```
Input Layer: (8015,)
  ├─ TF-IDF Features: 8000 (vocabulary size)
  └─ Phishing Features: 15 (URL patterns, keywords, character analysis)

Hidden Layers:
  ├─ Dense(512, relu) + Dropout(0.3)
  ├─ Dense(256, relu) + Dropout(0.3)
  └─ Dense(128, relu) + Dropout(0.2)

Output Layer: Dense(3, softmax)
```

**Total Parameters**: ~4.3 million (optimized with quantization)

## Training Details

- **Dataset**: Combined email, SMS, and URL datasets (80K+ samples)
- **Class Balance**: SMOTE applied (33% each class)
- **Train/Test Split**: 80/20 with stratification
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Early Stopping**: Patience=5 (restored best weights)
- **Training Duration**: 28 epochs

## Feature Engineering

The model combines TF-IDF features with phishing-specific features:

### TF-IDF Features (8000)
- Vocabulary size: 8000 words
- N-gram range: (1, 2) - unigrams and bigrams
- Min document frequency: 2

### Phishing Features (15)
1. URL detection (count, has_url)
2. IP address URLs (has_ip_url)
3. Suspicious TLDs (has_suspicious_tld)
4. URL shorteners (has_url_shortener)
5. HTTPS usage (has_https)
6. Urgency keywords (urgency_count)
7. Credential keywords (credential_count)
8. Money/prize keywords (money_count)
9. Caps ratio (caps_ratio)
10. Special character count
11. Exclamation marks
12. Question marks
13. Text length
14. Word count

## Usage in Flutter

### 1. Add Dependencies
```yaml
dependencies:
  tflite_flutter: ^0.10.0
```

### 2. Load Model
```dart
import 'package:tflite_flutter/tflite_flutter.dart';

// Load model
final interpreter = await Interpreter.fromAsset('assets/phishing_model.tflite');

// Load tokenizer
final tokenizer = await loadTokenizer('assets/tokenizer.json');

// Load labels
final labels = await loadLabels('assets/labels.json');
```

### 3. Preprocessing
You'll need to implement text preprocessing in Dart/Flutter:
1. Clean text (lowercase, remove special chars)
2. Tokenize using the vocabulary from `tokenizer.json`
3. Apply TF-IDF transformation using IDF values
4. Extract phishing features
5. Combine into input vector of size (8015,)

### 4. Run Inference
```dart
// Prepare input tensor (1 x 8015)
var input = [preprocessedFeatures];

// Prepare output tensor (1 x 3)
var output = List.filled(1 * 3, 0.0).reshape([1, 3]);

// Run inference
interpreter.run(input, output);

// Get prediction
int predictedClass = output[0].indexOf(output[0].reduce(max));
String label = labels[predictedClass]; // legitimate, phishing, or spam
double confidence = output[0][predictedClass];
```

## Retraining the Model

To retrain the model with updated data:

```bash
python3 model/train_tensorflow_model.py
```

This will:
1. Load datasets from `data/` directory
2. Apply preprocessing and feature extraction
3. Train the TensorFlow model
4. Convert to TensorFlow Lite
5. Save all artifacts to `flutter_model/` directory

## Model Validation

The model was validated on a held-out test set (20% of data) and achieved:
- Overall accuracy: 99.19%
- All classes above 98% F1-score
- ROC AUC: 0.999-1.000 for all classes
- Sample predictions: 100% confidence on test cases

## Requirements

### Training Requirements
- Python 3.8+
- TensorFlow 2.20+
- scikit-learn 1.5.2
- imbalanced-learn 0.14.0
- pandas, numpy, matplotlib, seaborn

### Flutter Requirements
- Flutter 3.0+
- tflite_flutter ^0.10.0
- Dart 2.17+

## Notes

- The model works 100% offline - no internet connection required
- TFLite format ensures fast inference on mobile devices
- Quantization applied for optimal size/speed tradeoff
- Model size is under 5MB for easy mobile deployment
- Input shape is fixed at (8015,) - ensure preprocessing matches

## Support

For issues or questions about the model:
1. Check model_info.json for metadata
2. Review confusion matrix and ROC curves for performance insights
3. Verify preprocessing matches training pipeline
4. Ensure TFLite interpreter version compatibility
