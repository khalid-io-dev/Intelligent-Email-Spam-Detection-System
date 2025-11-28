#!/usr/bin/env python3
"""
Example Client - Classification Service Consumer
BMSecurity Spam Detection System

Usage:
    python example_client.py

This script demonstrates how to interact with the FastAPI services.
"""

import requests
import json
from typing import List, Dict

# Configuration
NLP_SERVICE_URL = "http://localhost:8001"
CLASSIFICATION_SERVICE_URL = "http://localhost:8002"

def check_service_health(url: str, service_name: str) -> bool:
    """Vérifier la santé d'un service"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ {service_name} is running")
            return True
        else:
            print(f"✗ {service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to {service_name} at {url}")
        return False
    except Exception as e:
        print(f"✗ Error checking {service_name}: {e}")
        return False

def clean_text(text: str) -> Dict:
    """Nettoyer du texte via NLP Service"""
    try:
        response = requests.post(
            f"{NLP_SERVICE_URL}/clean",
            json={"text": text},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return None

def classify_email(text: str) -> Dict:
    """Classifier un email"""
    try:
        response = requests.post(
            f"{CLASSIFICATION_SERVICE_URL}/predict",
            json={"text": text},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error classifying email: {e}")
        return None

def batch_classify_emails(emails: List[str]) -> Dict:
    """Classifier plusieurs emails en batch"""
    try:
        emails_data = [{"text": email} for email in emails]
        response = requests.post(
            f"{CLASSIFICATION_SERVICE_URL}/batch-predict",
            json=emails_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in batch classification: {e}")
        return None

def main():
    """Fonction principale avec exemples"""
    
    print("=" * 70)
    print("BMSecurity - Spam Detection System - Client Example")
    print("=" * 70)
    
    # 1. Vérifier les services
    print("\n1️⃣  Checking Services Health...")
    print("-" * 70)
    nlp_ok = check_service_health(NLP_SERVICE_URL, "NLP Service (8001)")
    class_ok = check_service_health(CLASSIFICATION_SERVICE_URL, "Classification Service (8002)")
    
    if not (nlp_ok and class_ok):
        print("\n❌ Services not available!")
        print("Please start the services using:")
        print("  Windows: start_services.bat")
        print("  Linux:   bash start_services.sh")
        return
    
    # 2. Test du nettoyage de texte
    print("\n2️⃣  Testing Text Cleaning...")
    print("-" * 70)
    test_text = "Hello!!! This is a TEST email from BMSecurity..."
    print(f"Original: {test_text}")
    cleaned = clean_text(test_text)
    if cleaned:
        print(f"Cleaned:  {cleaned['cleaned']}")
    
    # 3. Exemples de classification
    print("\n3️⃣  Testing Email Classification...")
    print("-" * 70)
    
    # Email légitime
    legitimate_email = """
    Hi John,
    
    I'm following up on our meeting yesterday about the Q4 project.
    Could you please send me the updated timeline?
    
    Thanks,
    Sarah
    """
    
    print(f"Email 1: {legitimate_email[:50]}...")
    result = classify_email(legitimate_email)
    if result:
        print(f"  Prediction: {result['prediction'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
    
    # Email spam
    spam_email = """
    CLICK HERE!!! You've WON a FREE iPhone!!!!
    Just verify your account at: http://bit.ly/scam
    ACT NOW! Limited time offer!!!
    """
    
    print(f"\nEmail 2: {spam_email[:50]}...")
    result = classify_email(spam_email)
    if result:
        print(f"  Prediction: {result['prediction'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
    
    # 4. Classification en batch
    print("\n4️⃣  Testing Batch Classification...")
    print("-" * 70)
    
    batch_emails = [
        "Meeting tomorrow at 10am",
        "CLICK HERE FOR FREE MONEY!!!",
        "The project status is on track",
        "LIMITED TIME OFFER - BUY NOW!!!",
        "Could you review the attached document?"
    ]
    
    results = batch_classify_emails(batch_emails)
    if results:
        print(f"Classified {len(results['predictions'])} emails:")
        for i, pred in enumerate(results['predictions'], 1):
            status = "✓ HAM" if pred['prediction'] == 'ham' else "✗ SPAM"
            print(f"  {i}. {status:8} ({pred['confidence']:.2%}) - {pred['text'][:30]}...")
    
    # 5. Résumé
    print("\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Review the GUIDE_COMPLET.md for more details")
    print("  2. Check the API documentation:")
    print("     - NLP Service:        http://localhost:8001/docs")
    print("     - Classification:     http://localhost:8002/docs")
    print("  3. Integrate the services into your application")

if __name__ == "__main__":
    main()
