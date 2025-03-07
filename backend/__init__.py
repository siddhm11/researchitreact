# backend/__init__.py

# Import key modules to make them available at the package level
from . import database
from . import models

# You can also import specific classes if you want them directly accessible
# from .models.research_recommender import ResearchRecommender

# Define what gets imported with "from backend import *"
__all__ = [
    'database',
    'models',
    # 'ResearchRecommender',  # Uncomment if you want this directly available
]