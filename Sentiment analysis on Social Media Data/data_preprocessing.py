"""
Data Preprocessing Module for Sentiment Analysis
This module contains functions for cleaning and preprocessing social media text data.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')



class TextPreprocessor:
    """Simple text preprocessing class for social media data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords so they are preserved
        negation_words = {'not', 'no', 'nor', 'never', "n't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 
                          'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
                          'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
                          "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                          'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
        self.stop_words = self.stop_words - negation_words
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean and preprocess a single text string
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text, remove #)
        text = re.sub(r'#', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """
        Tokenize and lemmatize text
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Processed text
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def preprocess(self, text, already_cleaned=False):
        """
        Complete preprocessing pipeline
        
        Args:
            text (str): Raw text
            already_cleaned (bool): If True, skip heavy cleaning (for pre-cleaned datasets)
            
        Returns:
            str: Fully preprocessed text
        """
        if already_cleaned:
            # For already cleaned data (like Reddit/Twitter datasets),
            # just do light preprocessing
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = ' '.join(text.split())  # Remove extra whitespace
            return text
        else:
            # Full preprocessing for raw text
            text = self.clean_text(text)
            text = self.tokenize_and_lemmatize(text)
            return text
    
    def preprocess_dataframe(self, df, text_column, already_cleaned=False):
        """
        Preprocess all texts in a dataframe column
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the column containing text
            already_cleaned (bool): If True, skip heavy cleaning
            
        Returns:
            pd.DataFrame: Dataframe with preprocessed text
        """
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(lambda x: self.preprocess(x, already_cleaned))
        return df


def load_sample_data():
    """
    Create sample social media data for demonstration
    
    Returns:
        pd.DataFrame: Sample dataset with text and sentiment labels
    """
    data = {
        'text': [
            "I love this product! It's amazing and works perfectly! 😊",
            "This is the worst experience ever. Totally disappointed.",
            "The service was okay, nothing special but not bad either.",
            "Absolutely fantastic! Best purchase I've made this year!",
            "Terrible quality. Would not recommend to anyone.",
            "It's fine. Does what it's supposed to do.",
            "Outstanding customer service! Very happy with my purchase!",
            "Complete waste of money. Very unhappy.",
            "Average product. Nothing to complain about.",
            "Excellent! Exceeded all my expectations!",
            "Poor quality and bad customer support.",
            "It's decent. Not great but acceptable.",
            "Amazing experience! Will definitely buy again!",
            "Horrible. Regret buying this.",
            "Neutral feelings about this product.",
            "Best thing ever! Highly recommend!",
            "Disappointing and overpriced.",
            "It works as expected. No complaints.",
            "Superb quality and fast delivery!",
            "Not satisfied at all. Very poor.",
            "Good value for money! Happy with it!",
            "Worst purchase ever made.",
            "It's alright. Nothing extraordinary.",
            "Fantastic product! Love it so much!",
            "Bad experience. Will not buy again."
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'neutral', 'positive',
            'negative', 'neutral', 'positive', 'negative', 'neutral',
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'positive', 'negative', 'neutral', 'positive', 'negative'
        ]
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the preprocessing
    print("Testing Text Preprocessor...")
    
    preprocessor = TextPreprocessor()
    
    sample_text = "I love this product! Check it out at http://example.com @user #awesome 😊"
    cleaned = preprocessor.preprocess(sample_text)
    
    print(f"\nOriginal: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test with sample data
    print("\n\nLoading sample data...")
    df = load_sample_data()
    print(f"Loaded {len(df)} samples")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
