# Spam Detection System - BMSecurity
## Système Intelligent de Détection de Spams

Une solution complète d'analyse et de classification d'emails spam utilisant NLP et Machine Learning.

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Installation](#installation)
3. [Utilisation](#utilisation)
4. [Architecture](#architecture)
5. [Résultats](#résultats)
6. [API Services](#api-services)

## 🎯 Vue d'ensemble

Ce projet implémente un système de détection de spams avec:
- **Analyse de données** complète
- **Prétraitement NLP** avancé (tokenization, stemming, stopwords)
- **Vectorisation** TF-IDF
- **Modèles ML** multiples (Naive Bayes, Logistic Regression, SVM)
- **Services FastAPI** pour déploiement en production

## 📦 Installation

### Dépendances
```bash
pip install -r requirements.txt
# ou pour API services
pip install -r requirements_api.txt
```

### Fichiers Nécessaires
- `DataSet_Emails.csv` - Dataset d'emails

## 🚀 Utilisation

### 1. Analyse Complète (Jupyter Notebook)
```bash
jupyter notebook spam_detection_analysis.ipynb
```

Ce notebook contient:
- Exploration des données
- Analyse de qualité des données
- Visualisations (WordClouds)
- Prétraitement du texte
- Entraînement des modèles
- Comparaison des performances

### 2. Script Standalone
```bash
python spam_detection.py
```

Génère:
- `class_distribution.png` - Distribution spam/ham
- `wordcloud_spam.png` - Mots fréquents spam
- `wordcloud_ham.png` - Mots fréquents ham
- `confusion_matrix_tfidf.png` - Performance du modèle
- `preprocessed_emails.csv` - Dataset préprocessé

### 3. Services FastAPI

#### NLP Service
```bash
uvicorn nlp_service:app --host 0.0.0.0 --port 8001
```

Endpoints:
- `POST /clean` - Nettoie le texte
- `POST /tokenize` - Tokenise le texte
- `POST /stem` - Effectue stemming
- `GET /health` - Santé du service

#### Classification Service
```bash
uvicorn classification_service:app --host 0.0.0.0 --port 8002
```

Endpoints:
- `POST /predict` - Prédit spam/ham
- `POST /batch-predict` - Prédictions batch
- `GET /health` - Santé du service
- `GET /info` - Informations du service

## 🏗️ Architecture

### Pipeline NLP
```
Email Input
    ↓
Lowercase Normalization
    ↓
Remove Punctuation (regex)
    ↓
Tokenization
    ↓
Remove Stopwords
    ↓
Stemming (Porter)
    ↓
TF-IDF Vectorization
    ↓
ML Classifier
    ↓
Spam/Ham Prediction
```

### Modèles Disponibles
- **Naive Bayes** - Rapide, baseline
- **Logistic Regression** - Équilibré, fiable
- **SVM Linear** - Performance élevée

## 📊 Résultats

### Données Statistiques
- Dataset: ~5000 emails
- Spam/Ham: Distribution analysée
- Train/Test: 80/20 split

### Métriques de Performance
Les modèles sont évalués sur:
- Accuracy (Précision globale)
- Precision (Vrais positifs / Tous positifs)
- Recall (Vrais positifs / Vrais labels)
- F1-Score (Moyenne harmonique)

## 💾 Fichiers Générés

```
ntlk/
├── spam_detection.py              # Script principal
├── spam_detection_analysis.ipynb   # Notebook d'analyse
├── nlp_service.py                 # API NLP
├── classification_service.py       # API Classification
├── requirements.txt               # Dépendances
├── requirements_api.txt           # Dépendances API
├── ARCHITECTURE.md                # Documentation
├── ml_models/                     # Modèles sauvegardés
│   ├── naive_bayes_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── svm_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_results.csv
└── preprocessed_dataset.csv       # Dataset préprocessé
```

## 🔧 Configuration

### Paramètres TF-IDF
- `max_features`: 3000
- `min_df`: 2
- `max_df`: 0.8

### Paramètres Modèles
- **Logistic Regression**: max_iter=1000
- **SVM**: max_iter=2000
- **Naive Bayes**: Default parameters

