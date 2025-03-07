import sqlite3
import datetime
import logging
import json
import numpy as np
from typing import Union

from database.db_connector import DBConnector
from .citations_fetcher import CitationsFetcher

logger = logging.getLogger(__name__)

class PaperQualityAssessor:
    """
    Assesses the quality of research papers based on various metrics.
    """
    
    def __init__(self, citations_fetcher: CitationsFetcher = None, db_connector=None):
        # Use provided DB connector or create a new one
        self.db_connector = db_connector or DBConnector()
        
        # Use provided citation fetcher or create a new one with our db_connector
        self.citation_fetcher = citations_fetcher or CitationsFetcher(db_connector=self.db_connector)
        
        # For normalization of scores across different metrics
        self.max_citation_count = 1000  # Will be updated as we process papers
        
    def assess_paper_quality(self, paper: dict) -> float:
        """
        Calculate a quality score for a paper
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        # Get citation metrics
        citation_metrics = self.citation_fetcher.get_citation_count(
            paper_id=paper['paper_id'] if 'paper_id' in paper else paper['id'],
            title=paper['title']
        )
        
        # Basic quality score components
        citation_score = self._normalize_citation_count(citation_metrics['citation_count'])
        recency_score = self._calculate_recency_score(paper['published'])
        
        # Consider venue quality if available
        venue_score = 0.0
        if 'venue' in citation_metrics and citation_metrics['venue']:
            venue_score = self._assess_venue_quality(citation_metrics['venue'])
        
        # Consider author impact
        author_score = 0.0
        if 'authors' in paper:
            # Handle authors in various formats
            authors_list = []
            if isinstance(paper['authors'], str):
                try:
                    authors_list = json.loads(paper['authors'])
                except json.JSONDecodeError:
                    # If it's not a JSON string, try comma-separated format
                    authors_list = [a.strip() for a in paper['authors'].split(',')]
            elif isinstance(paper['authors'], list):
                authors_list = paper['authors']
            
            if authors_list:
                author_metrics = self.citation_fetcher.get_author_impact(authors_list)
                # Scale author score based on h-index and normalize between 0-1
                max_h_index = author_metrics.get('max_h_index', 0)
                avg_h_index = author_metrics.get('avg_h_index', 0)
                
                # Use a weighted combination of max and average h-index
                weighted_h_index = 0.7 * max_h_index + 0.3 * avg_h_index
                
                # Normalize: h-index of 40+ is considered very high (1.0 score)
                author_score = min(weighted_h_index / 40, 1.0)
        
        # Analysis of paper content
        content_score = self._analyze_paper_content(paper)
        
        if recency_score > 0.8:  
            weights = {
                'citation': 0.2,
                'recency': 0.3,
                'venue': 0.2,
                'author': 0.15,
                'content': 0.15
            }
        else: 
            weights = {
                'citation': 0.5,
                'recency': 0.1,
                'venue': 0.2,
                'author': 0.1,
                'content': 0.1
            }
        
        # Consider influential citations more heavily if available
        if 'influential_citations' in citation_metrics and citation_metrics['influential_citations'] > 0:
            # Normalize influential citations (these are more important than raw counts)
            influential_ratio = citation_metrics['influential_citations'] / max(1, citation_metrics['citation_count'])
            influential_score = min(1.0, influential_ratio * 2)  # Scale up: 50%+ influential is top score
            
            # Adjust citation score to include influential citations factor
            citation_score = 0.7 * citation_score + 0.3 * influential_score
        
        # Calculate final weighted score
        quality_score = (
            weights['citation'] * citation_score +
            weights['recency'] * recency_score +
            weights['venue'] * venue_score +
            weights['author'] * author_score +
            weights['content'] * content_score
        )
        
        # Add bonus for papers with high citation count that are also recent
        if citation_score > 0.7 and recency_score > 0.7:
            quality_score += 0.1
            
        # Add bonus for papers from top authors in top venues
        if author_score > 0.8 and venue_score > 0.8:
            quality_score += 0.05
        
        # Ensure score is between 0 and 1
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Log the score and its components
        logger.info(f"Quality assessment for {paper.get('paper_id', paper.get('id', 'unknown'))}: " +
                    f"Final={quality_score:.2f}, " +
                    f"Citation={citation_score:.2f}, " +
                    f"Recency={recency_score:.2f}, " +
                    f"Venue={venue_score:.2f}, " +
                    f"Author={author_score:.2f}, " +
                    f"Content={content_score:.2f}")
        
        return quality_score
    
    def _normalize_citation_count(self, citation_count: int) -> float:
        """Normalize citation count to a score between 0 and 1"""
        # Update maximum citation count seen so far for better normalization
        self.max_citation_count = max(self.max_citation_count, citation_count)
        
        # Use logarithmic scaling to handle papers with vastly different citation counts
        if citation_count == 0:
            return 0.0
        log_citations = np.log1p(citation_count)
        log_max = np.log1p(self.max_citation_count)
        
        return log_citations / log_max
    
    def _calculate_recency_score(self, published_date: Union[str, datetime.date]) -> float:
        """Calculate recency score (newer papers score higher)"""
        if isinstance(published_date, str):
            try:
                published_date = datetime.datetime.strptime(published_date, "%Y-%m-%d").date()
            except ValueError:
                published_date = datetime.datetime.now().date()
                
        days_old = (datetime.datetime.now().date() - published_date).days
        
        # Papers less than 6 months old get high scores
        if days_old < 180:
            return 0.8 + (180 - days_old) / 900  # Max 1.0 for very recent papers
        
        # Papers between 6 months and 3 years get moderate scores
        elif days_old < 1095:
            return 0.4 + (1095 - days_old) / 2300
            
        # Papers older than 3 years get lower scores
        else:
            return max(0.1, 0.4 - (days_old - 1095) / 10000)
    
    def update_quality_scores_in_db(self, paper_ids=None):
        """
        Update quality scores for papers in the database
        
        Args:
            paper_ids: List of paper IDs to update, or None for all papers
        """
        # Connect to both databases
        papers_conn = self.db_connector.get_research_papers_connection()
        citations_conn = self.db_connector.get_citations_connection()
        
        papers_cursor = papers_conn.cursor()
        citations_cursor = citations_conn.cursor()
        
        # Get papers to update
        if paper_ids:
            placeholders = ', '.join(['?'] * len(paper_ids))
            papers_cursor.execute(f"SELECT * FROM papers WHERE paper_id IN ({placeholders})", paper_ids)
        else:
            papers_cursor.execute("SELECT * FROM papers")
            
        papers_data = papers_cursor.fetchall()
        column_names = [description[0] for description in papers_cursor.description]
        logger.info(f"Updating quality scores for {len(papers_data)} papers")
        
        # Update scores
        for paper_row in papers_data:
            # Convert to dict format expected by assess_paper_quality
            paper = {column_names[i]: paper_row[i] for i in range(len(column_names))}
            paper_id = paper['paper_id']
            
            # Calculate quality score
            quality_score = self.assess_paper_quality(paper)
            
            # Update in database
            citations_cursor.execute(
                "UPDATE citations SET quality_score = ? WHERE paper_id = ?",
                (quality_score, paper_id)
            )
        
        # Commit changes and close
        citations_conn.commit()
        papers_conn.close()
        citations_conn.close()
        
        logger.info(f"Updated quality scores for {len(papers_data)} papers")
    
    def _assess_venue_quality(self, venue: str) -> float:
        """Assess quality of publication venue"""
        # This is a simplified implementation
        # In a real system, you would have a database of journal/conference rankings
        
        # Check for top-tier venues (examples from CS/ML)
        top_venues = {
            'NeurIPS': 1.0,
            'ICML': 1.0,
            'ICLR': 1.0,
            'CVPR': 1.0,
            'ECCV': 0.95,
            'ACL': 0.95,
            'EMNLP': 0.9,
            'JMLR': 0.95,
            'TPAMI': 0.95,
            'Nature': 1.0,
            'Science': 1.0,
            'Cell': 0.95
        }
        
        # Check if venue matches or contains a top venue name
        for top_venue, score in top_venues.items():
            if top_venue.lower() == venue.lower() or top_venue.lower() in venue.lower():
                return score
                
        return 0.3
    
    def _analyze_paper_content(self, paper: dict) -> float:
        """Analyze paper content for quality indicators"""
        score = 0.5  # Default score
        
        # Check for presence of abstract
        if 'abstract' in paper and paper['abstract'] and len(paper['abstract']) > 100:
            # Abstract length and complexity can indicate paper quality
            abstract_length = len(paper['abstract'])
            if abstract_length > 1500:
                score += 0.15
            elif abstract_length > 800:
                score += 0.1
            elif abstract_length > 400:
                score += 0.05
                
            # Check for key quality indicators in abstract
            quality_indicators = [
                'novel', 'state-of-the-art', 'state of the art', 'sota',
                'outperform', 'improve', 'contribution', 'breakthrough',
                'demonstrate', 'experiment', 'dataset'
            ]
            
            abstract_lower = paper['abstract'].lower()
            indicator_count = sum(1 for indicator in quality_indicators if indicator in abstract_lower)
            score += min(indicator_count * 0.02, 0.1)
            
            # Penalize for vague language (might indicate lower quality)
            vague_terms = ['may', 'might', 'could', 'possibly', 'perhaps', 'potential']
            vague_count = sum(1 for term in vague_terms if f" {term} " in f" {abstract_lower} ")
            score -= min(vague_count * 0.02, 0.1)
            
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))