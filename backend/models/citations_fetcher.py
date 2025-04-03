import logging
import json
import requests
import time
from typing import Dict, List, Any, Optional
from database.mongo_connector import MongoConnector

logger = logging.getLogger(__name__)

class CitationsFetcher:
    """
    Fetches citation information for research papers from various sources.
    Uses MongoDB for storage.
    """
    
    def __init__(self, mongo_connector=None):
        """
        Initialize CitationsFetcher with MongoDB connection.
        
        Args:
            mongo_connector: MongoDB connector instance
        """
        if mongo_connector is None:
            self.mongo_connector = MongoConnector()
        else:
            self.mongo_connector = mongo_connector
            
        self.citations_collection = self.mongo_connector.get_citations_collection()
        self.authors_collection = self.mongo_connector.get_authors_collection()
        
        # Create indexes
        self.citations_collection.create_index("paper_id", unique=True)
        self.authors_collection.create_index("author_id", unique=True)
        self.authors_collection.create_index("name")
    
    def get_citation_count(self, paper_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Get citation count for a paper from MongoDB or external API.
        """
        # First check if we have recent data in our database
        citation_data = self.citations_collection.find_one({"paper_id": paper_id})
        
        if citation_data:
            # Remove MongoDB _id field from the result
            if "_id" in citation_data:
                del citation_data["_id"]
            return citation_data
        
        # Attempt to fetch real citation data from Semantic Scholar
        citation_data = self.fetch_real_citation_data(paper_id)
        
        if citation_data:
            citation_count = citation_data.get("citationCount", 0)
            influential_citations = citation_data.get("influentialCitationCount", 0)
            venue = citation_data.get("venue", "")
        else:
            # Fallback: use simulation if real data not available
            citation_count = self._simulate_citation_count(paper_id, title)
            influential_citations = int(citation_count * 0.3)
            venue = self._extract_venue_from_title(title) if title else ""
        
        # Store the retrieved data in MongoDB
        citation_doc = {
            "paper_id": paper_id,
            "citation_count": citation_count,
            "influential_citations": influential_citations,
            "venue": venue,
            "quality_score": 0.5,  # Default quality score
            "last_updated": time.time()
        }
        
        self.citations_collection.update_one(
            {"paper_id": paper_id},
            {"$set": citation_doc},
            upsert=True
        )
        
        return citation_doc
    
    def get_author_impact(self, authors: List[str]) -> Dict[str, Any]:
        """
        Get impact metrics for a list of authors
        
        Args:
            authors: List of author names
            
        Returns:
            Dictionary with author impact metrics
        """
        h_indices = []
        
        for author in authors:
            # Check if we have this author in MongoDB
            author_doc = self.authors_collection.find_one({"name": author})
            
            if author_doc:
                h_indices.append(author_doc["h_index"])
            else:
                # Simulate h-index based on author name
                h_index = self._simulate_h_index(author)
                h_indices.append(h_index)
                
                # Store in MongoDB
                author_id = author.replace(" ", "_").lower()
                self.authors_collection.update_one(
                    {"author_id": author_id},
                    {"$set": {
                        "author_id": author_id,
                        "name": author,
                        "h_index": h_index,
                        "total_citations": 0,  # Default value
                        "last_updated": time.time()
                    }},
                    upsert=True
                )
        
        if not h_indices:
            return {"max_h_index": 0, "avg_h_index": 0, "authors_count": 0}
        
        return {
            "max_h_index": max(h_indices),
            "avg_h_index": sum(h_indices) / len(h_indices),
            "authors_count": len(authors)
        }
    
    # The _simulate_citation_count, _extract_venue_from_title, _simulate_h_index, 
    # and fetch_real_citation_data methods remain the same

    def _simulate_citation_count(self, paper_id: str, title: Optional[str] = None) -> int:
        """
        Simulate citation count based on paper ID and title
        In a real implementation, this would call an external API
        
        Args:
            paper_id: ID of the paper
            title: Title of the paper
            
        Returns:
            Simulated citation count
        """
        # Use paper_id hash for deterministic but seemingly random results
        import hashlib
        
        # Create a hash from paper_id
        hash_obj = hashlib.md5(paper_id.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Base citation count on hash
        base_citation = hash_int % 1000
        
        # If title is available, adjust based on title length and keywords
        if title:
            # Longer titles might indicate more comprehensive papers
            title_factor = min(len(title) / 50, 2.0)
            
            # Check for keywords that might indicate higher impact
            impact_keywords = ["novel", "state-of-the-art", "framework", "survey", "review"]
            keyword_factor = 1.0
            for keyword in impact_keywords:
                if keyword.lower() in title.lower():
                    keyword_factor += 0.5
            
            return int(base_citation * title_factor * keyword_factor)
        
        return base_citation
    
    def _extract_venue_from_title(self, title: str) -> str:
        """
        Extract potential venue information from title
        In a real implementation, this would be more sophisticated
        
        Args:
            title: Title of the paper
            
        Returns:
            Extracted venue or empty string
        """
        # List of common venues
        venues = ["NIPS", "NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "CVPR", "ECCV", "ICCV", "KDD", "WWW", "SIGIR"]
        
        for venue in venues:
            if venue in title:
                return venue
        
        return ""
    
    
    def _simulate_h_index(self, author: str) -> int:
        """
        Simulate h-index for an author
        In a real implementation, this would call an external API
        
        Args:
            author: Author name
            
        Returns:
            Simulated h-index
        """
        # Use author name hash for deterministic but seemingly random results
        import hashlib
        
        # Create a hash from author name
        hash_obj = hashlib.md5(author.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Base h-index on hash (between 1 and 50)
        return (hash_int % 50) + 1
    
    def fetch_real_citation_data(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch real citation data from Semantic Scholar for a given paper.
        The paper_id is assumed to be an arXiv ID.
        """
        try:
            # Construct Semantic Scholar API ID for arXiv papers
            api_id = f"ARXIV:{paper_id}"
            url = f"https://api.semanticscholar.org/graph/v1/paper/{api_id}?fields=citationCount,influentialCitationCount,venue"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning("Semantic Scholar API returned status code: {}".format(response.status_code))
                return None
        except Exception as e:
            logger.error("Error fetching citation data from Semantic Scholar: {}".format(e))
            return None


