import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from indicnlp.tokenize import indic_tokenize
from unidecode import unidecode
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import os
import joblib

class NepaliTextProcessor:
    def __init__(self):
        # Romanized Nepali normalization rules
        self.roman_mapping = {
            r'\bjai\b': 'jay', r'\bshree\b': 'shri', r'\bsri\b': 'shri',
            r'\bchha\b': 'cha', r'\bchh\b': 'ch', r'\bvayo\b': 'bhayo',
            r'\bhunxa\b': 'huncha', r'\bgarna\b': 'garne', r'\bparxa\b': 'parcha',
            r'\bkaam\b': 'kam', r'\bpaani\b': 'pani', r'\bbhayo\b': 'bhayo',
            r'\bthiyo\b': 'thiyo', r'\bhuda\b': 'huda', r'\bma\b': 'maa'
        }
        
        # Emoticon sentiment mapping
        self.emoticon_sentiment = {
            '❤️': 2, '🙏': 2, '🌹': 1, '👍': 1, '🎉': 1, '😍': 2,
            '🌺': 1, '🏵️': 1, '👏': 1, '🎊': 1, '🎇': 1, '💐': 1,
            '🎁': 1, '❣️': 2, '🥰': 2, '🤩': 1, '😔': -1, '😡': -2,
            '🤔': -1, '😛': -1, '😝': -1, '😅': -1, '😂': -1, '😆': -1,
            '😹': -1, '🥴': -1, '🤣': -1, '😮': -1, '😤': -2, '😒': -1,
            '😠': -2, '🇳🇵': 0, '📱': 0, '🏦': 0
        }
        
        # Nepali stopwords (extended)
        self.nepali_stopwords = set([
            "छ", "छन्", "छु", "गरे", "गरी", "गर्न", "गर्छ", "गर्दछ", "गर्नु", "गर्ने", 
            "हो", "हुन्", "हुन", "भन्ने", "भने", "भन्नु", "र", "तर", "वा", "अथवा", 
            "यो", "त्यो", "यस", "उस", "यी", "ती", "यहाँ", "त्यहाँ", "कहाँ", "जहाँ", 
            "के", "की", "का", "को", "कि", "कै", "कान", "कानी", "है", "हरु", "हरू", 
            "आफ्नो", "आफू", "म", "हामी", "तिमी", "उनी", "उनले", "उनको", "तपाईं", 
            "तिनीहरू", "हाम्रो", "तपाईंको", "उनीहरूको", "अरु", "अन्य", "एक", "दुई", 
            "तीन", "चार", "पाँच", "धेरै", "केही", "सबै", "कुनै", "कता", "कती", 
            "कसरी", "किन", "जब", "तब", "पनि", "मात्र", "बाहेक", "सम्म", "लाई", "ले", 
            "बाट", "देखि", "सँग", "विरुद्ध", "नजिक", "तल", "माथि", "भित्र", "बाहिर"
        ])
        
        self.english_stopwords = set(stopwords.words('english'))

    def safe_text(self, text):
        """Ensure text is string and not empty"""
        if pd.isna(text) or text is None:
            return ""
        return str(text).strip()

    def extract_emoticons(self, text):
        """Extract and classify emoticons while preserving original text"""
        text = self.safe_text(text)
        if not text:
            return [], [], 0, 'neutral', ""
            
        # Find all emojis
        emoticons = [c for c in text if c in emoji.EMOJI_DATA]
        clean_text = text
        
        # Remove emojis from text (keeping Nepali intact)
        for e in emoticons:
            clean_text = clean_text.replace(e, ' ')
        
        clean_text = ' '.join(clean_text.split())
        
        # Classify emoticons
        emoticon_info = []
        sentiment_score = 0
        
        for e in set(emoticons):
            score = self.emoticon_sentiment.get(e, 0)
            emoticon_info.append(f"{e}:{score}")
            sentiment_score += score
        
        emoticon_sentiment = 'positive' if sentiment_score > 0 else \
                            'negative' if sentiment_score < 0 else 'neutral'
        
        return emoticons, emoticon_info, emoticon_sentiment, sentiment_score, clean_text

    def normalize_roman_nepali(self, text):
        """Only normalize Romanized parts, preserving Nepali"""
        text = self.safe_text(text)
        if not text:
            return ""
            
        # Split text into Nepali and non-Nepali parts
        parts = re.split('([\u0900-\u097F]+)', text)
        
        for i in range(len(parts)):
            if not re.search('[\u0900-\u097F]', parts[i]):
                # Process non-Nepali parts
                parts[i] = unidecode(parts[i])
                for pattern, replacement in self.roman_mapping.items():
                    parts[i] = re.sub(pattern, replacement, parts[i], flags=re.IGNORECASE)
                parts[i] = re.sub(r'(\w)\1+', r'\1', parts[i])
        
        return ''.join(parts)

    def clean_text(self, text):
        """Clean while preserving Nepali characters"""
        text = self.safe_text(text)
        if not text:
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Remove special characters except Nepali and basic punctuation
        text = re.sub(r'[^\w\s\u0900-\u097F.,!?]', ' ', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def detect_language(self, text):
        """Enhanced language detection"""
        text = self.safe_text(text)
        if not text:
            return 'unknown'
            
        has_nepali = re.search(r'[\u0900-\u097F]', text)
        has_english = re.search(r'[a-zA-Z]', text)
        
        if has_nepali and has_english:
            return 'mixed'
        elif has_nepali:
            return 'nepali'
        elif has_english:
            return 'english'
        return 'unknown'

    def tokenize_text(self, text, lang):
        """Language-aware tokenization"""
        text = self.safe_text(text)
        if not text:
            return []
            
        if lang == 'nepali':
            return indic_tokenize.trivial_tokenize(text)
        elif lang == 'english':
            return word_tokenize(text.lower())
        else:  # mixed or unknown
            tokens = []
            for part in re.split('([\u0900-\u097F]+)', text):
                if re.search('[\u0900-\u097F]', part):
                    tokens.extend(indic_tokenize.trivial_tokenize(part))
                else:
                    tokens.extend(word_tokenize(part.lower()))
            return tokens

    def remove_stopwords(self, tokens, lang):
        """Language-specific stopword removal"""
        if not tokens:
            return []
            
        if lang == 'nepali':
            return [word for word in tokens if word not in self.nepali_stopwords]
        elif lang == 'english':
            return [word for word in tokens if word not in self.english_stopwords]
        else:  # mixed
            return [word for word in tokens 
                   if word not in self.nepali_stopwords and 
                   word not in self.english_stopwords]

    def analyze_sentiment(self, text):
        """Analyze text sentiment using TextBlob (for English text)"""
        text = self.safe_text(text)
        if not text:
            return 'neutral', 0
        
        # First check if text is Nepali
        if self.detect_language(text) == 'nepali':
            # Simple keyword-based approach for Nepali
            positive_words = ['राम्रो', 'उत्कृष्ट', 'धन्यवाद', 'माया', 'सुन्दर']
            negative_words = ['खराब', 'नराम्रो', 'घिन', 'गाली', 'नापसंद']
            
            positive_count = sum(1 for word in text.split() if word in positive_words)
            negative_count = sum(1 for word in text.split() if word in negative_words)
            
            if positive_count > negative_count:
                return 'positive', 1
            elif negative_count > positive_count:
                return 'negative', -1
            else:
                return 'neutral', 0
        else:
            # Use TextBlob for English/mixed text
            try:
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity
                
                if polarity > 0.1:
                    return 'positive', 1
                elif polarity < -0.1:
                    return 'negative', -1
                else:
                    return 'neutral', 0
            except:
                return 'neutral', 0

    def process_row(self, row):
        """Complete row processing pipeline"""
        try:
            # Ensure Sentiment is string
            original_sentiment = str(row['Sentiment']) if pd.notna(row['Sentiment']) else ""
            
            # First extract emoticons and get clean text
            emoticons, emoticon_info, emoticon_sentiment, emoticon_score, clean_text = \
                self.extract_emoticons(row['Commnets'])
            
            # Normalize Romanized parts while preserving Nepali
            normalized_text = self.normalize_roman_nepali(clean_text)
            
            # Final cleaning
            final_text = self.clean_text(normalized_text)
            
            # Detect language
            language = self.detect_language(final_text)
            
            # Tokenize
            tokens = self.tokenize_text(final_text, language)
            
            # Remove stopwords
            filtered_tokens = self.remove_stopwords(tokens, language)
            
            # Analyze text sentiment
            text_sentiment, text_score = self.analyze_sentiment(final_text)
            
            # Combine emoticon and text sentiment
            combined_score = emoticon_score + text_score
            combined_sentiment = 'positive' if combined_score > 0 else \
                               'negative' if combined_score < 0 else 'neutral'
            
            return pd.Series({
                'original_text': row['Commnets'],
                'cleaned_text': final_text,
                'language': language,
                'tokens': tokens,
                'filtered_tokens': filtered_tokens,
                'emoticons': ' '.join(emoticons),
                'emoticon_info': ' | '.join(emoticon_info),
                'emoticon_sentiment': emoticon_sentiment,
                'emoticon_score': emoticon_score,
                'text_sentiment': text_sentiment,
                'text_score': text_score,
                'combined_sentiment': combined_sentiment,
                'combined_score': combined_score,
                'original_sentiment': original_sentiment
            })
        except Exception as e:
            print(f"Error processing row: {e}")
            return pd.Series({
                'original_text': row['Commnets'] if 'Commnets' in row else "",
                'cleaned_text': "",
                'language': "unknown",
                'tokens': [],
                'filtered_tokens': [],
                'emoticons': "",
                'emoticon_info': "",
                'emoticon_sentiment': "neutral",
                'emoticon_score': 0,
                'text_sentiment': "neutral",
                'text_score': 0,
                'combined_sentiment': "neutral",
                'combined_score': 0,
                'original_sentiment': str(row['Sentiment']) if 'Sentiment' in row else ""
            })
