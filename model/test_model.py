#!/usr/bin/env python3
"""Test script to verify the trained phishing detection model"""

import os
import pickle
import numpy as np
from train_text_classifier import clean_text, extract_phishing_features, create_feature_matrix
import pandas as pd
from scipy.sparse import hstack

# Load model artifacts
MODEL_PATH = os.path.join("model", "rf_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

def load_model():
    """Load trained model and preprocessors"""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, vectorizer, label_encoder

def predict_message(text, model, vectorizer, label_encoder):
    """Predict class for a single message"""
    # Create dataframe with single message
    df = pd.DataFrame([{
        'text': text,
        'original_text': text
    }])
    
    # Clean text
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
    
    # Get class name and confidence
    predicted_class = label_encoder.classes_[prediction]
    confidence = probabilities[prediction] * 100
    
    return predicted_class, confidence, probabilities

def main():
    print("="*80)
    print("🧪 Testing Phishing Detection Model")
    print("="*80)
    
    # Load model
    print("\n📥 Loading model...")
    model, vectorizer, label_encoder = load_model()
    print(f"   ✓ Model loaded")
    print(f"   ✓ Classes: {label_encoder.classes_}")
    
    # Test messages
    test_messages = [
        # Legitimate messages
        ("Hey, let's meet for coffee tomorrow at 3pm", "legitimate"),
        ("Your package from Amazon has been delivered", "legitimate"),
        ("Meeting reminder: Team standup at 10am", "legitimate"),
        
        # Spam messages
        ("CONGRATULATIONS! You've WON $1,000,000! Claim your prize now!", "spam"),
        ("FREE iPhone! Click here to get yours today!!!", "spam"),
        ("Make money fast! Work from home $5000/week!", "spam"),
        
        # Phishing messages
        ("URGENT: Your bank account has been suspended. Verify your password immediately at http://fake-bank.tk/login", "phishing"),
        ("Action required: Your Apple ID will be locked. Confirm your account details: http://apple-verify.xyz", "phishing"),
        ("Security alert: Unusual login detected. Reset your password now: http://192.168.1.1/reset", "phishing"),
    ]
    
    print("\n🔮 Testing predictions:\n")
    
    correct = 0
    high_confidence = 0
    
    for text, expected in test_messages:
        predicted_class, confidence, probs = predict_message(text, model, vectorizer, label_encoder)
        
        is_correct = predicted_class == expected
        is_high_conf = confidence >= 80
        
        if is_correct:
            correct += 1
        if is_high_conf:
            high_confidence += 1
        
        status = "✓" if is_correct else "✗"
        conf_status = "🎯" if is_high_conf else "⚠️"
        
        print(f"{status} {conf_status} Expected: {expected:12s} | Predicted: {predicted_class:12s} | Confidence: {confidence:.1f}%")
        print(f"   Text: {text[:70]}...")
        
        # Show all probabilities
        prob_str = " | ".join([f"{label_encoder.classes_[i]}: {probs[i]*100:.1f}%" for i in range(len(probs))])
        print(f"   Probabilities: {prob_str}")
        print()
    
    print("="*80)
    print(f"📊 Results:")
    print(f"   Accuracy: {correct}/{len(test_messages)} ({correct/len(test_messages)*100:.1f}%)")
    print(f"   High Confidence (>80%): {high_confidence}/{len(test_messages)} ({high_confidence/len(test_messages)*100:.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
