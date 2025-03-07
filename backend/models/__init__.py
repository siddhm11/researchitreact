# Import classes to make them available at package level
from .text_preprocessor import TextPreprocessor
from .citations_fetcher import CitationsFetcher
from .paper_quality_assessor import PaperQualityAssessor
from .arxiv_fetcher import ArxivFetcher
from .embedding_system import EmbeddingSystem

# Define what gets imported with "from models import *"
__all__ = [
    'TextPreprocessor',
    'CitationsFetcher',
    'PaperQualityAssessor',
    'ArxivFetcher',
    'EmbeddingSystem'
]