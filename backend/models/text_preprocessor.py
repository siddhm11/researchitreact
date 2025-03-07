import re
from typing import List
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool, cpu_count

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Advanced text preprocessing for research papers.
    Implements efficient cleaning, tokenization, and normalization.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add research-specific stopwords
        research_stopwords = {'et', 'al', 'fig', 'figure', 'table', 'eq', 'equation', 'ref'}
        self.stop_words.update(research_stopwords)
        self.lemmatizer = WordNetLemmatizer()
        # Cache for lemmatized words to avoid redundant processing
        self.lemma_cache = {}
        
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace"""
        if not text:
            return ""
            
        # Replace line breaks and tabs with spaces
        text = re.sub(r'[\n\t\r]', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove LaTeX equations (often between $ symbols)
        text = re.sub(r'\$+[^$]+\$+', ' equation ', text)
        
        # Remove citations like [1], [2,3], etc.
        text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
        
        # Remove redundant spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def cached_lemmatize(self, word: str) -> str:
        """Lemmatize with caching for performance"""
        if word not in self.lemma_cache:
            self.lemma_cache[word] = self.lemmatizer.lemmatize(word)
        return self.lemma_cache[word]
    
    def process_text(self, text: str, lemmatize: bool = True) -> str:
        """Process text with tokenization, stopword removal, and optional lemmatization"""
        if not text:
            return ""
            
        # Clean the text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        filtered_tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        
        # Apply lemmatization if requested
        if lemmatize:
            # Only lemmatize tokens that are likely nouns or informative terms
            # This selective approach balances precision and performance
            if len(filtered_tokens) > 100:  # For long texts, be selective
                processed_tokens = []
                for token in filtered_tokens:
                    if len(token) > 3:  # Focus on longer words that are more likely to be significant
                        processed_tokens.append(self.cached_lemmatize(token))
                    else:
                        processed_tokens.append(token)
                return ' '.join(processed_tokens)
            else:
                return ' '.join([self.cached_lemmatize(t) for t in filtered_tokens])
        
        return ' '.join(filtered_tokens)
    
    def batch_process(self, texts: List[str], lemmatize: bool = True, n_jobs: int = None) -> List[str]:
        """Process multiple texts in parallel"""
        if not n_jobs:
            n_jobs = max(1, cpu_count() - 1)  # Use all cores except one by default
            
        # For small batches, don't use multiprocessing overhead
        if len(texts) < 10:
            return [self.process_text(text, lemmatize) for text in texts]
            
        with Pool(n_jobs) as pool:
            return pool.starmap(self.process_text, [(text, lemmatize) for text in texts])