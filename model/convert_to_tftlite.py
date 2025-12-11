"""
Convert scikit-learn Random Forest model to TensorFlow Lite
"""

import pickle
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import json

# Paths
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "rf_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

OUTPUT_DIR = Path("flutter_model")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("🔄 Converting scikit-learn model to TensorFlow Lite")
print("="*70)

# Load scikit-learn model
print("\n📂 Loading scikit-learn model...")
with open(MODEL_PATH, 'rb') as f:
    rf_model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

print(f"   ✓ Model loaded:  {type(rf_model)}")
print(f"   ✓ Classes: {rf_model.classes_}")
print(f"   ✓ Number of trees: {rf_model.n_estimators}")
print(f"   ✓ Number of features: {rf_model. n_features_in_}")

# Save vectorizer vocabulary for Flutter
print("\n📝 Exporting TF-IDF vocabulary...")
vocabulary = vectorizer.get_feature_names_out().tolist()
vocab_dict = {word: idx for idx, word in enumerate(vocabulary)}

vocab_path = OUTPUT_DIR / "vocabulary.json"
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"   ✓ Vocabulary saved:  {len(vocabulary)} words → {vocab_path}")

# Save IDF values
idf_values = vectorizer.idf_. tolist()
idf_path = OUTPUT_DIR / "idf_values.json"
with open(idf_path, 'w') as f:
    json.dump(idf_values, f)

print(f"   ✓ IDF values saved → {idf_path}")

# Save label mapping
label_mapping = {
    "classes":  rf_model.classes_. tolist(),
    "num_classes": len(rf_model.classes_)
}
label_path = OUTPUT_DIR / "labels.json"
with open(label_path, 'w') as f:
    json.dump(label_mapping, f, indent=2)

print(f"   ✓ Label mapping saved → {label_path}")


# ============================================
# Convert Random Forest to TensorFlow
# ============================================

print("\n🔄 Converting Random Forest to TensorFlow...")

# Since Random Forest can't be directly converted to TFLite,
# we'll create a TensorFlow model that mimics the RF behavior
# OR we'll export the trees as a lookup table

# Option A: Save all tree predictions as a lookup (not practical for large models)
# Option B: Convert to a neural network approximation
# Option C: Use TF Decision Forests

# For this project, we'll use **TF Decision Forests**

try:
    import tensorflow_decision_forests as tfdf
    
    print("   ℹ️  Using TensorFlow Decision Forests for conversion")
    
    # This is complex - let's use a simpler approach
    # We'll save the model parameters and implement prediction in Dart
    
except ImportError: 
    print("   ⚠️  tensorflow_decision_forests not installed")
    print("   ℹ️  We'll use alternative approach:  Export model parameters")


# ============================================
# Alternative: Export Random Forest Parameters
# ============================================

print("\n📦 Exporting Random Forest parameters...")

# Extract all trees
trees_data = []

for tree_idx, tree in enumerate(rf_model.estimators_):
    tree_structure = {
        'tree_id': tree_idx,
        'feature':  tree.tree_.feature. tolist(),
        'threshold':  tree.tree_.threshold.tolist(),
        'children_left': tree.tree_.children_left.tolist(),
        'children_right': tree.tree_. children_right.tolist(),
        'value': tree.tree_.value.tolist(),
        'n_node_samples': tree.tree_.n_node_samples.tolist()
    }
    trees_data.append(tree_structure)

# This creates a VERY large file - let's save just metadata
metadata = {
    'n_estimators': rf_model.n_estimators,
    'n_classes': len(rf_model.classes_),
    'n_features':  rf_model.n_features_in_,
    'max_depth': rf_model.max_depth,
}

metadata_path = OUTPUT_DIR / "model_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"   ✓ Model metadata saved → {metadata_path}")

# For production:  Save first 10 trees as example
# Full model is too large for mobile
sample_trees = trees_data[:10]
trees_path = OUTPUT_DIR / "sample_trees.json"
with open(trees_path, 'w') as f:
    json.dump(sample_trees, f, indent=2)

print(f"   ✓ Sample trees saved (10/{len(trees_data)}) → {trees_path}")


# ============================================
# Better Approach: Use ONNX
# ============================================

print("\n🔄 Converting to ONNX format (better for mobile)...")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, rf_model.n_features_in_]))]
    
    # Convert
    onnx_model = convert_sklearn(rf_model, initial_types=initial_type, target_opset=12)
    
    # Save ONNX model
    onnx_path = OUTPUT_DIR / "phishing_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"   ✓ ONNX model saved → {onnx_path}")
    print(f"   ✓ Model size: {onnx_path.stat().st_size / 1024:.2f} KB")
    
    # Now convert ONNX to TFLite
    print("\n🔄 Converting ONNX to TensorFlow Lite...")
    
    # This requires onnx-tf package
    import onnx
    from onnx_tf. backend import prepare
    
    onnx_model_loaded = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model_loaded)
    
    # Export to TensorFlow SavedModel
    tf_model_path = OUTPUT_DIR / "tf_model"
    tf_rep.export_graph(str(tf_model_path))
    
    print(f"   ✓ TensorFlow model saved → {tf_model_path}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
    converter.optimizations = [tf.lite.Optimize. DEFAULT]
    tflite_model = converter.convert()
    
    tflite_path = OUTPUT_DIR / "phishing_model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"   ✓ TFLite model saved → {tflite_path}")
    print(f"   ✓ TFLite size: {len(tflite_model) / 1024:.2f} KB")
    
    print("\n" + "="*70)
    print("✅ Conversion Complete!")
    print("="*70)
    print(f"\n📁 Output files in:  {OUTPUT_DIR}/")
    print("   - phishing_model.tflite   ← Use this in Flutter")
    print("   - vocabulary.json          ← TF-IDF vocabulary")
    print("   - idf_values.json          ← IDF weights")
    print("   - labels.json              ← Class labels")
    print("\n🚀 Ready to integrate with Flutter!")
    
except ImportError as e:
    print(f"   ⚠️  Missing dependencies: {e}")
    print("\n📦 Install required packages:")
    print("   pip install skl2onnx onnx onnx-tf")
    print("\n   Then run this script again.")

except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n   Falling back to pickle export...")
    
    # Fallback:  Just copy the pickle files for Flutter to use via platform channels
    import shutil
    
    shutil.copy(MODEL_PATH, OUTPUT_DIR / "rf_model.pkl")
    shutil.copy(VECTORIZER_PATH, OUTPUT_DIR / "vectorizer. pkl")
    shutil.copy(LABEL_ENCODER_PATH, OUTPUT_DIR / "label_encoder.pkl")
    
    print(f"   ✓ Models copied to {OUTPUT_DIR}/")
    print("   ⚠️  You'll need to use Platform Channels to call Python from Flutter")

print("\n" + "="*70)