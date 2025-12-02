"""
Sentiment Analysis Model Training Script
This script trains multiple models and saves the best performing one.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import TextPreprocessor, load_sample_data


class SentimentAnalysisModel:
    """Simple sentiment analysis model trainer"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=30, max_depth=20, random_state=42, n_jobs=-1)
        }
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0
    
    def prepare_data(self, df, text_column='text', label_column='sentiment', already_cleaned=False):
        """
        Prepare data for training
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            label_column (str): Name of label column
            already_cleaned (bool): If True, skip heavy cleaning
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("Preprocessing text data...")
        df = self.preprocessor.preprocess_dataframe(df, text_column, already_cleaned)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Split data
        X = df['cleaned_text']
        y = df[label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Vectorize text
        print("\nVectorizing text using TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and select the best one
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
        """
        print("\n" + "="*60)
        print("Training Models")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Save best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "="*60)
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print("="*60)
        
        return results
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path='confusion_matrix.png'):
        """
        Plot and save confusion matrix
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['negative', 'neutral', 'positive'],
                    yticklabels=['negative', 'neutral', 'positive'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
    
    def save_model(self, model_path='sentiment_model.pkl', 
                   vectorizer_path='vectorizer.pkl',
                   preprocessor_path='preprocessor.pkl'):
        """
        Save the trained model and preprocessing objects
        
        Args:
            model_path: Path to save the model
            vectorizer_path: Path to save the vectorizer
            preprocessor_path: Path to save the preprocessor
        """
        print("\nSaving model and preprocessing objects...")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"Model saved to {model_path}")
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {vectorizer_path}")
        
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        print(f"Preprocessor saved to {preprocessor_path}")
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'accuracy': self.best_accuracy,
            'classes': ['negative', 'neutral', 'positive']
        }
        
        with open('model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        print("Model info saved to model_info.pkl")
    
    def predict(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Predicted sentiment
        """
        # Preprocess
        cleaned = self.preprocessor.preprocess(text)
        
        # Vectorize
        vectorized = self.vectorizer.transform([cleaned])
        
        # Predict
        prediction = self.best_model.predict(vectorized)[0]
        
        return prediction


def main():
    """Main training function"""
    print("="*60)
    print("Sentiment Analysis Model Training")
    print("="*60)
    
    # Load data from Reddit and Twitter datasets
    print("\nLoading Reddit and Twitter datasets...")
    
    try:
        # Load Reddit data
        df_reddit = pd.read_csv('Reddit_Data.csv')
        print(f"Loaded Reddit data: {len(df_reddit)} samples")
        
        # Load Twitter data
        df_twitter = pd.read_csv('Twitter_Data.csv')
        print(f"Loaded Twitter data: {len(df_twitter)} samples")
        
        # Standardize column names
        df_reddit = df_reddit.rename(columns={'clean_comment': 'text'})
        df_twitter = df_twitter.rename(columns={'clean_text': 'text'})
        
        # Map numeric categories to sentiment labels
        # -1 = negative, 0 = neutral, 1 = positive
        category_map = {-1: 'negative', -1.0: 'negative', 
                       0: 'neutral', 0.0: 'neutral',
                       1: 'positive', 1.0: 'positive'}
        
        df_reddit['sentiment'] = df_reddit['category'].map(category_map)
        df_twitter['sentiment'] = df_twitter['category'].map(category_map)
        
        # Remove any rows with missing values
        df_reddit = df_reddit.dropna(subset=['text', 'sentiment'])
        df_twitter = df_twitter.dropna(subset=['text', 'sentiment'])
        
        # Combine both datasets
        df = pd.concat([df_reddit[['text', 'sentiment']], 
                       df_twitter[['text', 'sentiment']]], 
                      ignore_index=True)
        
        print(f"\nCombined dataset: {len(df)} samples")
        print(f"\nSentiment distribution:")
        print(df['sentiment'].value_counts())
        
    except FileNotFoundError as e:
        print(f"\nError: Dataset files not found!")
        print(f"Please make sure 'Reddit_Data.csv' and 'Twitter_Data.csv' are in the current directory.")
        print(f"\nFalling back to sample data...")
        df = load_sample_data()
        print(f"Loaded {len(df)} samples from sample data")
    
    # Initialize model
    model = SentimentAnalysisModel()
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(df, already_cleaned=True)
    
    # Train models
    results = model.train_models(X_train, X_test, y_train, y_test)
    
    # Plot confusion matrix
    y_pred = model.best_model.predict(X_test)
    model.plot_confusion_matrix(y_test, y_pred)
    
    # Save model
    model.save_model()
    
    # Test prediction
    print("\n" + "="*60)
    print("Testing Predictions")
    print("="*60)
    
    test_texts = [
        "This is absolutely amazing! I love it!",
        "Terrible experience. Very disappointed.",
        "It's okay, nothing special."
    ]
    
    for text in test_texts:
        prediction = model.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {prediction}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
