"""Utility functions for phishing detection model"""

import re
from urllib.parse import urlparse

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
        except (ValueError, TypeError):
            # Skip malformed URLs (ValueError for invalid IPv6, TypeError for None/invalid types)
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
