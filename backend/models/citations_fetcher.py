import sqlite3
import requests
import time
import json
import re
import datetime
import logging
from typing import List
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from database.db_connector import DBConnector


logger = logging.getLogger(__name__)

class CitationsFetcher:
    """
    Fetches citation information for papers to determine their impact and quality.
    """
    
    def __init__(self, db_connector=None):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.citation_cache = {}
        
        # Use provided DB connector or create a new one
        self.db_connector = db_connector or DBConnector()
        self.conn = self.db_connector.get_citations_connection()
        
        self.create_citation_table()
        self.create_author_table()
    
    def create_citation_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS citations (
            paper_id TEXT PRIMARY KEY,
            citation_count INTEGER,
            influential_citations INTEGER,
            venue TEXT,
            last_updated TEXT,
            h_index REAL,
            journal_impact REAL,
            quality_score REAL DEFAULT 0.0
        )"""  # Add closing parenthesis
        self.conn.execute(query)
        self.conn.commit()

    def create_author_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS author_metrics (
            authors_key TEXT PRIMARY KEY,
            max_h_index REAL,
            avg_h_index REAL,
            total_citations INTEGER,
            last_updated TEXT
        )"""  # Add closing parenthesis
        self.conn.execute(query)
        self.conn.commit()
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_citation_count(self, paper_id: str, title: str = None) -> dict:
        """
        Get citation count and other metrics for a paper
        
        Args:
            paper_id: arXiv ID of the paper
            title: Title of the paper (for fallback search)
            
        Returns:
            Dictionary with citation metrics
        """
        # Check cache first
        if paper_id in self.citation_cache:
            return self.citation_cache[paper_id]

        # Check database next
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM citations WHERE paper_id = ?", (paper_id,))
        result = cursor.fetchone()
        
        if result:
            # Convert database row to dictionary
            column_names = [description[0] for description in cursor.description]
            metrics = {column_names[i]: result[i] for i in range(len(column_names))}
            
            # Add to cache
            self.citation_cache[paper_id] = metrics
            logger.info(f"Retrieved citation data from database for {paper_id}")
            return metrics
                
        metrics = {
            'paper_id': paper_id,
            'citation_count': 0,
            'h_index': 0,
            'journal_impact': 0.0,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        # Try to get citation info from Semantic Scholar
        try:
            url = f"https://api.semanticscholar.org/v1/paper/arXiv:{paper_id}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                metrics['citation_count'] = data.get('citationCount', 0)
                # Store the influentialCitationCount as a bonus metric if available
                if 'influentialCitationCount' in data:
                    metrics['influential_citations'] = data['influentialCitationCount']
                    
                # Store venue information if available
                if 'venue' in data and data['venue']:
                    metrics['venue'] = data['venue']
                    
                # Store in database
                cursor.execute("""
                    INSERT OR REPLACE INTO citations 
                    (paper_id, citation_count, influential_citations, venue, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    paper_id, 
                    metrics['citation_count'],
                    metrics.get('influential_citations', 0),
                    metrics.get('venue', ''),
                    metrics['last_updated']
                ))
                self.conn.commit()
                    
                # Store in cache
                self.citation_cache[paper_id] = metrics
                logger.info(f"Retrieved citation data for {paper_id}: {metrics['citation_count']} citations")
                return metrics
        except Exception as e:
            logger.warning(f"Error fetching citation data from Semantic Scholar for {paper_id}: {e}")
            
        # Fallback to Google Scholar if title is provided
        if title:
            try:
                # Wait to avoid rate limiting
                time.sleep(2)
                
                # Search Google Scholar using the paper title
                search_query = title.replace(' ', '+')
                url = f"https://scholar.google.com/scholar?q={search_query}"
                
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Look for citation information
                    for result in soup.select('.gs_ri'):
                        result_title = result.select_one('.gs_rt')
                        if result_title and self._title_similarity(title, result_title.text.strip()) > 0.8:
                            citation_info = result.select_one('.gs_fl')
                            if citation_info:
                                citation_text = citation_info.text
                                citation_match = re.search(r'Cited by (\d+)', citation_text)
                                if citation_match:
                                    metrics['citation_count'] = int(citation_match.group(1))
                                    logger.info(f"Retrieved citation data from Google Scholar for {paper_id}: {metrics['citation_count']} citations")
                                    
                                    # Store in database
                                    cursor.execute("""
                                        INSERT OR REPLACE INTO citations 
                                        (paper_id, citation_count, last_updated)
                                        VALUES (?, ?, ?)
                                    """, (
                                        paper_id, 
                                        metrics['citation_count'],
                                        metrics['last_updated']
                                    ))
                                    self.conn.commit()
                                    break
            except Exception as e:
                logger.warning(f"Error fetching citation data from Google Scholar for {paper_id}: {e}")
        
        # Store in cache and database (even if we couldn't find citations)
        self.citation_cache[paper_id] = metrics
        
        # Make sure we store the empty result too
        if metrics['citation_count'] == 0:
            cursor.execute("""
                INSERT OR REPLACE INTO citations 
                (paper_id, citation_count, last_updated)
                VALUES (?, ?, ?)
            """, (
                paper_id, 
                metrics['citation_count'],
                metrics['last_updated']
            ))
            self.conn.commit()
            
        return metrics
    
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two paper titles"""
        # Simple implementation using character-level similarity
        title1 = title1.lower()
        title2 = title2.lower()
        
        # Remove common prefixes/suffixes and special characters
        for prefix in ['the ', 'a ', 'an ']:
            if title1.startswith(prefix):
                title1 = title1[len(prefix):]
            if title2.startswith(prefix):
                title2 = title2[len(prefix):]
                
        title1 = re.sub(r'[^\w\s]', '', title1)
        title2 = re.sub(r'[^\w\s]', '', title2)
        
        # Check if one is substring of another
        if title1 in title2 or title2 in title1:
            return 0.9
            
        # Calculate Jaccard similarity of words
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def __del__(self):
        """Close database connection when object is destroyed"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def get_author_impact(self, authors: List[str]) -> dict:
        """
        Get impact metrics for paper authors
        
        Args:
            authors: List of author names
            
        Returns:
            Dictionary with author impact metrics
        """
        # Create a key to identify this set of authors
        authors_key = "-".join(sorted(authors))
        
        # Check database first
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM author_metrics WHERE authors_key = ?", (authors_key,))
            result = cursor.fetchone()
            
            if result:
                # Convert database row to dictionary
                column_names = [description[0] for description in cursor.description]
                metrics = {column_names[i]: result[i] for i in range(len(column_names))}
                
                # Check if data is recent (within 30 days)
                last_updated = datetime.datetime.fromisoformat(metrics['last_updated'])
                days_since_update = (datetime.datetime.now() - last_updated).days
                
                if days_since_update < 30:
                    logger.info(f"Using cached author metrics for {authors_key}")
                    return metrics
        except Exception as e:
            logger.warning(f"Error querying author metrics from database: {e}")
        
        # Default metrics
        impact_metrics = {
            'authors_key': authors_key,
            'max_h_index': 0,
            'avg_h_index': 0,
            'total_citations': 0,
            'last_updated': datetime.datetime.now().isoformat()
        }
        
        # Fetch metrics for each author
        h_indices = []
        total_citations = 0
        
        for author in authors:
            try:
                # Normalize author name
                author_name = author.strip().lower()
                
                # Try Semantic Scholar API first
                url = f"https://api.semanticscholar.org/v1/author/search?query={author_name}"
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') and len(data['data']) > 0:
                        # Get the first (most relevant) author
                        author_data = data['data'][0]
                        author_id = author_data.get('authorId')
                        
                        if author_id:
                            # Get detailed author data
                            author_url = f"https://api.semanticscholar.org/v1/author/{author_id}"
                            author_response = requests.get(author_url, headers=self.headers, timeout=10)
                            
                            if author_response.status_code == 200:
                                detailed_data = author_response.json()
                                h_index = detailed_data.get('hIndex', 0)
                                citations = detailed_data.get('citationCount', 0)
                                
                                h_indices.append(h_index)
                                total_citations += citations
                                
                                # Sleep to avoid rate limiting
                                time.sleep(1)
                
                # Fallback method: try to estimate h-index from recent publications
                # This would require additional implementation in a production system
                
            except Exception as e:
                logger.warning(f"Error fetching impact data for author {author}: {e}")
        
        # Update metrics based on what we found
        if h_indices:
            impact_metrics['max_h_index'] = max(h_indices)
            impact_metrics['avg_h_index'] = sum(h_indices) / len(h_indices)
            impact_metrics['total_citations'] = total_citations
        
        # Store in database
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO author_metrics 
                (authors_key, max_h_index, avg_h_index, total_citations, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (
                authors_key,
                impact_metrics['max_h_index'],
                impact_metrics['avg_h_index'],
                impact_metrics['total_citations'],
                impact_metrics['last_updated']
            ))
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Error storing author metrics in database: {e}")
        
        return impact_metrics