"""
Research Recommender System
Integrates the ArXiv fetcher, embedding system, and quality assessor
to provide paper recommendations based on similarity and quality.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Optional, Union, Tuple
import time
import traceback

from database.mongo_connector import MongoConnector
from .text_preprocessor import TextPreprocessor
from .embedding_system import EmbeddingSystem
from .arxiv_fetcher import ArxivFetcher
from .citations_fetcher import CitationsFetcher
from .paper_quality_assessor import PaperQualityAssessor


logger = logging.getLogger("research_api.research_recommender")

class ResearchRecommender:
    def __init__(self):
        """Initialize the research recommender with all required components"""
        self.logger = logger
        self.logger.info("Initializing Research Recommender")
        
        # Create MongoDB connector first
        self.mongo_connector = MongoConnector()
        
        self.text_preprocessor = TextPreprocessor()
        self.embedding_system = EmbeddingSystem()
        self.fetcher = ArxivFetcher(mongo_connector=self.mongo_connector)
        self.citations_fetcher = CitationsFetcher(mongo_connector=self.mongo_connector)
        self.quality_assessor = PaperQualityAssessor(citations_fetcher=self.citations_fetcher)
        
        self._indexed_papers = set()
        self.logger.info("Research Recommender initialized")

    
    def load_index(self, index_path: str) -> bool:
        """
        Load an existing index from the specified path
        Args:
            index_path (str): Path to the directory containing index files
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            papers_file = os.path.join(index_path, "indexed_papers.txt")
            if os.path.exists(papers_file):
                with open(papers_file, 'r', encoding='utf-8') as f:
                    self._indexed_papers = set(line.strip() for line in f)
                self.logger.info(f"Loaded {len(self._indexed_papers)} indexed paper IDs")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            return False
    
    def save_index(self, index_path: str) -> bool:
        """
        Save the current index to the specified path
        Args:
            index_path (str): Path to the directory where index files will be saved
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            os.makedirs(index_path, exist_ok=True)
            
            papers_file = os.path.join(index_path, "indexed_papers.txt")
            with open(papers_file, 'w', encoding='utf-8') as f:
                for paper_id in self._indexed_papers:
                    f.write(f"{paper_id}\n")
            self.logger.info(f"Saved {len(self._indexed_papers)} indexed paper IDs")
            return True
        except Exception as e:
            self.logger.error(f"Error saving index: {str(e)}")
            return False
    
    def fetch_and_index(self, query: str, max_results: int = 50,
                        date_start: Optional[str] = None,
                        date_end: Optional[str] = None,
                        force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch papers based on query and index them for later retrieval.
        
        Args:
            query (str): ArXiv search query.
            max_results (int): Maximum number of results to return.
            date_start (str, optional): Start date in YYYY-MM-DD format.
            date_end (str, optional): End date in YYYY-MM-DD format.
            force_refresh (bool, optional): If True, forces re-fetching even if papers exist.

        Returns:
            pd.DataFrame: DataFrame containing the fetched papers.
        """
        try:
            self.logger.info(f"Fetching papers with query: {query}, max_results={max_results}")
            
            # 🔹 Force refresh: Clear indexed papers to avoid skipping results
            if force_refresh:
                self._indexed_papers.clear()  # ✅ Clears previous indexing
            
            # 🔹 Fetch new papers
            papers_df = self.fetcher.fetch_store(query=query, max_results=max_results, force_refresh=force_refresh)

            if papers_df.empty:
                self.logger.warning(f"No new papers found for query: {query}")
                return pd.DataFrame()

            # 🔹 Index only new papers
            self._index_papers(papers_df)

            return papers_df

        except Exception as e:
            self.logger.error(f"Error in fetch_and_index: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    
    def search_with_ranking(self, query: str, max_results: int = 50,
                        date_start: Optional[str] = None,
                        date_end: Optional[str] = None,
                        categories: Optional[List[str]] = None,
                        rank_by_citations: bool = False,
                        rank_by_quality: bool = False) -> pd.DataFrame:
        """
        Search for papers with advanced ranking options
        
        Args:
            query: ArXiv search query
            max_results: Maximum number of results to return
            date_start: Start date in YYYY-MM-DD format
            date_end: End date in YYYY-MM-DD format
            categories: List of arXiv categories to filter by
            rank_by_citations: Whether to rank by citation count
            rank_by_quality: Whether to rank by quality score
            
        Returns:
            DataFrame with search results
        """
        try:
            # Construct the query with categories if provided
            final_query = query
            if categories:
                cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
                final_query = f"({query}) AND ({cat_query})"
                
            # Fetch and index papers
            papers_df = self.fetch_and_index(
                query=final_query, 
                max_results=max_results,
                date_start=date_start,
                date_end=date_end,
                force_refresh=True
            )
            
            if papers_df.empty:
                self.logger.warning(f"No papers found for query: {final_query}")
                return pd.DataFrame()
                
            # Add paper IDs if not present
            paper_id_col = 'paper_id' if 'paper_id' in papers_df.columns else 'id'
            if 'paper_id' not in papers_df.columns:
                papers_df['paper_id'] = papers_df[paper_id_col]
                
            # Add ranking metrics if requested
            if rank_by_citations or rank_by_quality:
                self.logger.info(f"Adding ranking metrics for {len(papers_df)} papers")
                
                # Add citation counts and quality scores
                papers_df['citation_count'] = 0
                papers_df['quality_score'] = 0.0
                
                for i, row in papers_df.iterrows():
                    paper_id = row['paper_id']
                    
                    # Get citation info
                    citation_info = self.citations_fetcher.get_citation_count(paper_id, row.get('title'))
                    papers_df.at[i, 'citation_count'] = citation_info.get('citation_count', 0)
                    
                    # Calculate quality score
                    if rank_by_quality:
                        papers_df.at[i, 'quality_score'] = self.quality_assessor.assess_paper_quality(row.to_dict())
                
                # Apply ranking
                if rank_by_citations and rank_by_quality:
                    # Combined ranking: normalize and add both scores
                    max_citations = papers_df['citation_count'].max() if papers_df['citation_count'].max() > 0 else 1
                    papers_df['combined_score'] = (papers_df['citation_count'] / max_citations * 0.7) + (papers_df['quality_score'] * 0.3)
                    papers_df = papers_df.sort_values('combined_score', ascending=False)
                elif rank_by_citations:
                    papers_df = papers_df.sort_values('citation_count', ascending=False)
                elif rank_by_quality:
                    papers_df = papers_df.sort_values('quality_score', ascending=False)
                    
            return papers_df
            
        except Exception as e:
            self.logger.error(f"Error in search_with_ranking: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    def _index_papers(self, papers_df: pd.DataFrame) -> None:
        if papers_df.empty:
            return
        
        paper_id_col = 'paper_id' if 'paper_id' in papers_df.columns else 'id'
        to_index_df = papers_df[~papers_df[paper_id_col].isin(self._indexed_papers)]

        if to_index_df.empty:
            self.logger.info("All papers are already indexed")
            return  # ⛔ If all papers are already indexed, the next search will return nothing!

        self.logger.info(f"Indexing {len(to_index_df)} new papers")
        self.embedding_system.process_papers(to_index_df)

        for paper_id in to_index_df[paper_id_col]:
            self._indexed_papers.add(paper_id)
 
    def recommend(self, text: Optional[str] = None, paper_id: Optional[str] = None,
                  k: int = 5, min_date: Optional[str] = None,
                  max_date: Optional[str] = None, quality_aware: bool = True) -> pd.DataFrame:
        """
        Get paper recommendations based on text or a paper ID
        Args:
            text (str, optional): Text to find similar papers to
            paper_id (str, optional): Paper ID to find similar papers to
            k (int): Number of recommendations to return
            min_date (str, optional): Minimum date in YYYY-MM-DD format
            max_date (str, optional): Maximum date in YYYY-MM-DD format
            quality_aware (bool): Whether to use quality assessment in ranking
        Returns:
            pd.DataFrame: DataFrame containing recommended papers
        """
        try:
            if not text and not paper_id:
                raise ValueError("Either text or paper_id must be provided")
            
            filter_criteria = {}
            if min_date:
                filter_criteria['published'] = {'after': min_date}
            if max_date:
                if 'published' in filter_criteria:
                    filter_criteria['published']['before'] = max_date
                else:
                    filter_criteria['published'] = {'before': max_date}
            
            similar_papers = self.embedding_system.recommend(
                text=text,
                paper_id=paper_id,
                k=k,
                filter_criteria=filter_criteria,
                quality_assessor=self.quality_assessor if quality_aware else None
            )
            
            if similar_papers.empty:
                return pd.DataFrame()
            
            return similar_papers
            
        except Exception as e:
            self.logger.error(f"Error in recommend: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def assess_paper_quality(self, paper_id: str) -> float:
        """
        Get quality assessment for a specific paper
        Args:
            paper_id (str): Paper ID to assess
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            paper_df = self.fetcher.check_database()
            paper = paper_df[paper_df['paper_id'] == paper_id]
            
            if paper.empty:
                paper_df = self.fetcher.fetch(f"id:{paper_id}", max_results=1)
                if paper_df.empty:
                    return 0.5 
                paper = paper_df.iloc[0]
            else:
                paper = paper.iloc[0]
            
            paper_dict = paper.to_dict()
            
            quality_score = self.quality_assessor.assess_paper_quality(paper_dict)
            
            return quality_score
        except Exception as e:
            self.logger.error(f"Error assessing paper quality for {paper_id}: {str(e)}")
            return 0.5  
    
    def get_citation_info(self, paper_id: str) -> Dict:
        """
        Get citation information for a specific paper
        Args:
            paper_id (str): Paper ID to get citation info for
        Returns:
            Dict: Citation information
        """
        try:
            paper_df = self.fetcher.check_database()
            paper = paper_df[paper_df['paper_id'] == paper_id]
            title = None
            if not paper.empty:
                title = paper.iloc[0]['title']
            return self.citations_fetcher.get_citation_count(paper_id, title)
            
        except Exception as e:
            self.logger.error(f"Error getting citation info for {paper_id}: {str(e)}")
            return {}
    
    def find_seminal_papers(self, topic: str, max_results: int = 10) -> pd.DataFrame:
        """
        Find seminal (highly influential) papers on a topic
        Args:
            topic (str): Research topic to find seminal papers for
            max_results (int): Maximum number of papers to return
        Returns:
            pd.DataFrame: DataFrame containing seminal papers
        """
        try:
            papers_df = self.fetcher.search_seminal_papers(topic, max_results=max(50, max_results * 3))
            
            if papers_df.empty:
                return pd.DataFrame()
            
            self._index_papers(papers_df)
            self.logger.info(f"Getting citation info for {len(papers_df)} papers")
            papers_df['citation_count'] = get_citation_info()
            for i, row in papers_df.iterrows():
                paper_id = row['paper_id'] if 'paper_id' in row else row['id']
                citation_info = self.get_citation_info(paper_id)
                papers_df.at[i, 'citation_count'] = citation_info.get('citation_count', 0)
                
                papers_df.at[i, 'quality_score'] = self.quality_assessor.assess_paper_quality(row.to_dict())
            
            papers_df = papers_df.sort_values(by='citation_count', ascending=False)
            
            return papers_df.head(max_results)
            
        except Exception as e:
            self.logger.error(f"Error finding seminal papers: {str(e)}")
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
