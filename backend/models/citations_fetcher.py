import sqlite3
import logging
import json
import requests
import time
from typing import Dict, List, Any, Optional

from database.db_connector import DBConnector

logger = logging.getLogger(__name__)

class CitationsFetcher:
    """
    Fetches citation information for research papers from various sources.
    """
    
    def __init__(self, db_connector=None):
        # Use provided DB connector or create a new one
        self.db_connector = db_connector or DBConnector()
        # Initialize connection to citations database
        self.conn = self.db_connector.get_citations_connection()
        self.create_tables()
        
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Create citations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citations (
                paper_id TEXT PRIMARY KEY,
                citation_count INTEGER DEFAULT 0,
                influential_citations INTEGER DEFAULT 0,
                venue TEXT,
                quality_score REAL DEFAULT 0.5,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
        
        # Create authors table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS authors (
            author_id TEXT PRIMARY KEY,
            name TEXT,
            h_index INTEGER DEFAULT 0,
            total_citations INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create paper_authors table (many-to-many relationship)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id TEXT,
            author_id TEXT,
            PRIMARY KEY (paper_id, author_id)
        )
        ''')
        
        self.conn.commit()
    
    def get_citation_count(self, paper_id: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Get citation count for a paper from database or external API
        
        Args:
            paper_id: ID of the paper
            title: Title of the paper (optional, for better matching)
            
        Returns:
            Dictionary with citation information
        """
        # First check if we have recent data in our database
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT citation_count, influential_citations, venue, quality_score, last_updated FROM citations WHERE paper_id = ?", 
            (paper_id,)
        )
        result = cursor.fetchone()
        
        # If we have recent data (less than 7 days old), return it
        if result:
            citation_count, influential_citations, venue, quality_score, last_updated = result
            
            # For demonstration, return cached data
            # In production, you might want to refresh data older than a certain threshold
            return {
                "paper_id": paper_id,
                "citation_count": citation_count,
                "influential_citations": influential_citations,
                "venue": venue,
                "quality_score": quality_score
            }
        
        # If no cached data, we would normally fetch from an external API
        # For this example, we'll simulate citation data
        citation_count = self._simulate_citation_count(paper_id, title)
        influential_citations = int(citation_count * 0.3)  # Simulate that 30% are influential
        venue = self._extract_venue_from_title(title) if title else ""
        
        # Store in database
        cursor.execute(
            "INSERT OR REPLACE INTO citations (paper_id, citation_count, influential_citations, venue) VALUES (?, ?, ?, ?)",
            (paper_id, citation_count, influential_citations, venue)
        )
        self.conn.commit()
        
        return {
            "paper_id": paper_id,
            "citation_count": citation_count,
            "influential_citations": influential_citations,
            "venue": venue,
            "quality_score": quality_score
        }

    
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
            # Check if we have this author in our database
            cursor = self.conn.cursor()
            cursor.execute("SELECT h_index FROM authors WHERE name = ?", (author,))
            result = cursor.fetchone()
            
            if result:
                h_indices.append(result[0])
            else:
                # Simulate h-index based on author name
                h_index = self._simulate_h_index(author)
                h_indices.append(h_index)
                
                # Store in database
                cursor.execute(
                    "INSERT OR REPLACE INTO authors (author_id, name, h_index) VALUES (?, ?, ?)",
                    (author.replace(" ", "_").lower(), author, h_index)
                )
                self.conn.commit()
        
        if not h_indices:
            return {"max_h_index": 0, "avg_h_index": 0, "authors_count": 0}
        
        return {
            "max_h_index": max(h_indices),
            "avg_h_index": sum(h_indices) / len(h_indices),
            "authors_count": len(authors)
        }
    
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
