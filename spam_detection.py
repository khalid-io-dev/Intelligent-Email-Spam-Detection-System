import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# =============================================================================
# 1. DATA ANALYSIS
# =============================================================================

print("="*70)
print("STEP 1: DATA ANALYSIS")
print("="*70)

# Load dataset
df = pd.read_csv('DataSet_Emails.csv')

print("\n1.1 Dataset Structure:")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\n1.2 Data Types:")
print(df.dtypes)

print(f"\n1.3 Missing Values:")
print(df.isnull().sum())

print(f"\n1.4 Duplicate Rows:")
duplicates = df.duplicated().sum()
print(f"Number of duplicates: {duplicates}")

print(f"\n1.5 Class Distribution:")
print(df['label'].value_counts())
print("\nClass Distribution (%):")
print(df['label'].value_counts(normalize=True) * 100)

# Visualize class distribution
plt.figure(figsize=(8, 5))
df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of Spam vs Ham Emails')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300)
print("\n✓ Saved: class_distribution.png")
plt.close()

# =============================================================================
# 2. WORDCLOUDS
# =============================================================================

print("\n" + "="*70)
print("STEP 2: GENERATING WORDCLOUDS")
print("="*70)

# Separate spam and ham emails
spam_emails = df[df['label'] == 'spam']['text'].str.cat(sep=' ')
ham_emails = df[df['label'] == 'ham']['text'].str.cat(sep=' ')

# Generate WordCloud for Spam emails
print("\nGenerating WordCloud for SPAM emails...")
wordcloud_spam = WordCloud(width=800, height=400, 
                           background_color='white',
                           colormap='Reds',
                           max_words=100).generate(spam_emails)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.title('Most Frequent Words in SPAM Emails', fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_spam.png', dpi=300)
print("✓ Saved: wordcloud_spam.png")
plt.close()

# Generate WordCloud for Ham emails
print("Generating WordCloud for HAM emails...")
wordcloud_ham = WordCloud(width=800, height=400,
                          background_color='white',
                          colormap='Greens',
                          max_words=100).generate(ham_emails)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_ham, interpolation='bilinear')
plt.title('Most Frequent Words in HAM (Legitimate) Emails', fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('wordcloud_ham.png', dpi=300)
print("✓ Saved: wordcloud_ham.png")
plt.close()

# =============================================================================
# 3. TEXT PREPROCESSING
# =============================================================================

print("\n" + "="*70)
print("STEP 3: TEXT PREPROCESSING")
print("="*70)

# Create a copy for preprocessing
df_clean = df.copy()

print("\n3.1 Initial dataset size:", df_clean.shape)

# Remove duplicates
df_clean = df_clean.drop_duplicates()
print(f"3.2 After removing duplicates: {df_clean.shape}")

# Remove rows with empty or missing text
df_clean = df_clean[df_clean['text'].notna()]
df_clean = df_clean[df_clean['text'].str.strip() != '']
print(f"3.3 After removing empty text: {df_clean.shape}")

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Complete text preprocessing pipeline:
    1. Convert to lowercase
    2. Remove punctuation and special characters
    3. Tokenize
    4. Remove stopwords
    5. Apply stemming
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and apply stemming
    processed_tokens = [stemmer.stem(word) for word in tokens 
                       if word not in stop_words and len(word) > 2]
    
    # Join tokens back into a single string
    return ' '.join(processed_tokens)

print("\n3.4 Applying preprocessing pipeline...")
print("    - Normalizing to lowercase")
print("    - Removing punctuation and special characters")
print("    - Tokenizing")
print("    - Removing stopwords")
print("    - Applying stemming (Porter Stemmer)")

df_clean['processed_text'] = df_clean['text'].apply(preprocess_text)

print("\n✓ Preprocessing complete!")
print("\nExample of preprocessing:")
print(f"Original: {df_clean.iloc[0]['text'][:100]}...")
print(f"Processed: {df_clean.iloc[0]['processed_text'][:100]}...")

# =============================================================================
# 4. TEXT VECTORIZATION
# =============================================================================

print("\n" + "="*70)
print("STEP 4: TEXT VECTORIZATION")
print("="*70)

# Prepare data for vectorization
X = df_clean['processed_text']
y = df_clean['label']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nDataset split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Method 1: TF-IDF Vectorization
print("\n4.1 TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

# Method 2: Count Vectorization
print("\n4.2 Count Vectorization...")
count_vec = CountVectorizer(max_features=3000)
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)
print(f"Count Vectorizer feature matrix shape: {X_train_count.shape}")

# =============================================================================
# 5. MODEL TRAINING AND EVALUATION
# =============================================================================

print("\n" + "="*70)
print("STEP 5: MODEL TRAINING AND EVALUATION")
print("="*70)

# Train model with TF-IDF features
print("\n5.1 Training Naive Bayes classifier with TF-IDF...")
model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf, y_train)

# Predictions
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

# Evaluate
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f"\nTF-IDF Model Accuracy: {accuracy_tfidf*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tfidf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tfidf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['ham', 'spam'], 
            yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix - TF-IDF Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_tfidf.png', dpi=300)
print("\n✓ Saved: confusion_matrix_tfidf.png")
plt.close()

# Train model with Count Vectorizer features
print("\n5.2 Training Naive Bayes classifier with Count Vectorizer...")
model_count = MultinomialNB()
model_count.fit(X_train_count, y_train)

# Predictions
y_pred_count = model_count.predict(X_test_count)

# Evaluate
accuracy_count = accuracy_score(y_test, y_pred_count)
print(f"\nCount Vectorizer Model Accuracy: {accuracy_count*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_count))

# =============================================================================
# 6. SAVE PREPROCESSED DATA
# =============================================================================

print("\n" + "="*70)
print("STEP 6: SAVING PREPROCESSED DATA")
print("="*70)

df_clean.to_csv('preprocessed_emails.csv', index=False)
print("\n✓ Saved: preprocessed_emails.csv")

print("\n" + "="*70)
print("SPAM DETECTION SYSTEM - COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - class_distribution.png")
print("  - wordcloud_spam.png")
print("  - wordcloud_ham.png")
print("  - confusion_matrix_tfidf.png")
print("  - preprocessed_emails.csv")
print("\nAll tasks completed successfully!")
