from pymongo import MongoClient
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MongoConnector:
    """MongoDB connection handler for the research recommendation system."""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string
        """
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client["research_papers_db"]
            logger.info("MongoDB connection established.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_papers_collection(self):
        """Get the papers collection."""
        return self.db.papers
    
    def get_citations_collection(self):
        """Get the citations collection."""
        return self.db.citations
        
    def get_authors_collection(self):
        """Get the authors collection."""
        return self.db.authors
    
    def close(self):
        """Close the MongoDB connection."""
        if hasattr(self, 'client'):
            self.client.close()
