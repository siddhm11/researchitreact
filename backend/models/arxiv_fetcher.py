import pandas as pd
import json
import datetime
from typing import List, Optional
import arxiv
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from database.mongo_connector import MongoConnector

# Set up logger
logger = logging.getLogger(__name__)

class ArxivFetcher:
    """
    Fetches research papers from Arxiv API with enhanced filtering.
    Implements retry mechanism for reliability.
    Uses MongoDB for storage.
    """
    
    def __init__(self, mongo_connector=None):
        """
        Initialize ArxivFetcher with MongoDB connection.
        
        Args:
            mongo_connector: MongoDB connector instance
        """
        if mongo_connector is None:
            self.mongo_connector = MongoConnector()
        else:
            self.mongo_connector = mongo_connector
            
        self.papers_collection = self.mongo_connector.get_papers_collection()
        # Create index on paper_id for efficient lookups
        self.papers_collection.create_index("paper_id", unique=True)

    def __del__(self):
        """Close MongoDB connection when object is destroyed"""
        if hasattr(self, 'mongo_connector'):
            self.mongo_connector.close()

    def fetch_store(self, query="cat:cs.LG", max_results=100, force_refresh: bool = False):
        """Fetch papers and store only new ones in MongoDB."""
        
        # Step 1: Fetch new papers
        df = self.fetch(query=query, max_results=max_results)
        
        if df.empty:
            print("⚠️ No papers retrieved! Skipping storage.")
            print(f"Query used: {query}")
            return df

        # Rename 'id' column to 'paper_id' to match schema
        if 'id' in df.columns and 'paper_id' not in df.columns:
            df = df.rename(columns={'id': 'paper_id'})

        # Step 2: Convert DataFrame to list of dictionaries
        papers = df.to_dict('records')
        
        # Step 3: Store papers in MongoDB
        new_count = 0
        for paper in papers:
            try:
                # Add timestamp
                paper['added_date'] = datetime.datetime.now()
                
                # Use upsert to insert or update if force_refresh is True
                if force_refresh:
                    result = self.papers_collection.update_one(
                        {"paper_id": paper["paper_id"]},
                        {"$set": paper},
                        upsert=True
                    )
                    if result.upserted_id or result.modified_count > 0:
                        new_count += 1
                else:
                    # Only insert if it doesn't exist
                    result = self.papers_collection.update_one(
                        {"paper_id": paper["paper_id"]},
                        {"$setOnInsert": paper},
                        upsert=True
                    )
                    if result.upserted_id:
                        new_count += 1
                        
            except Exception as e:
                logger.error(f"Error storing paper {paper.get('paper_id')}: {e}")
        
        print(f"✅ Stored {new_count} new papers.")
        
        # Return DataFrame of only the new papers
        if new_count == 0:
            return pd.DataFrame()
        
        return df

    def check_database(self):
        """Retrieve stored papers as DataFrame"""
        papers = list(self.papers_collection.find({}, {'_id': 0}))
        df = pd.DataFrame(papers)
        
        if not df.empty:
            print(f"📄 Retrieved {len(df)} papers from MongoDB.")
        
        return df
    
    # The fetch() and other methods remain largely unchanged
    # Just keep the fetch(), search_by_keywords(), and search_seminal_papers() methods as they are
  
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch(self, query: str = "cat:cs.LG", max_results: int = 100, 
          date_start: Optional[str] = None, date_end: Optional[str] = None, 
          sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance) -> pd.DataFrame:
    
        logger.info(f"Fetching {max_results} papers for query: {query}")
        
        # Construct the query string with date filters if available
        if date_start and date_end:
            query += f' AND submittedDate:[{date_start} TO {date_end}]'

        client = arxiv.Client(page_size=100, delay_seconds=3)
        search = arxiv.Search(query=query, max_results=max_results, sort_by=sort_by)

        try:
            results = list(client.results(search))
            logger.info(f"Total papers fetched: {len(results)}")

            if not results:
                logger.warning("⚠️ No papers retrieved! Check query or network connection.")
                return pd.DataFrame()  # ✅ Return empty DataFrame instead of recursive call
     
# Debugging step to check if papers are being fetched


            papers = []
            for result in results:
                paper = {
                    'paper_id': result.entry_id.split('/')[-1],
                    'title': str(result.title).strip(),  # ✅ Ensure title is a string
                    'abstract': result.summary,
                    'authors': json.dumps([a.name for a in result.authors]),  # Convert to JSON string
                    'primary_category': result.primary_category,
                    'categories': json.dumps(result.categories),  # Convert to JSON string
                    'published': result.published.date().isoformat(),  # Convert to string
                    'pdf_url': result.pdf_url,
                    'source': 'arxiv'
                }

                papers.append(paper)

            df = pd.DataFrame(papers)
            logger.info(f"✅ Successfully converted {len(df)} papers into DataFrame.")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching papers: {e}")
            return pd.DataFrame()

    def search_by_keywords(self, 
                          keywords: List[str], 
                          categories: List[str] = None,
                          max_results: int = 100,
                          date_start: Optional[str] = None, 
                          date_end: Optional[str] = None) -> pd.DataFrame:
        """
        Search for papers by keywords and categories
        
        Args:
            keywords: List of keywords to search for
            categories: List of arXiv categories (e.g. ['cs.LG', 'cs.AI'])
            max_results: Maximum number of papers to fetch
            date_start: Start date in format YYYY-MM-DD
            date_end: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame with search results
        """
        # Build keyword part of query
        keyword_query = " AND ".join([f"all:{kw}" for kw in keywords])
        if not keywords or not isinstance(keywords, list):
            logger.error("Keywords must be a non-empty list")
            return pd.DataFrame()
    
        if max_results <= 0 or max_results > 1000:
            logger.warning(f"Invalid max_results: {max_results}. Using default of 100.")
            max_results = 100
            
        # Validate date format
        if date_start:
            try:
                datetime.datetime.strptime(date_start, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid date_start format: {date_start}. Using None.")
                date_start = None
                
        if date_end:
            try:
                datetime.datetime.strptime(date_end, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid date_end format: {date_end}. Using None.")
                date_end = None
        # Add categories if specified
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({keyword_query}) AND ({cat_query})"
        else:
            query = keyword_query
            
        return self.fetch(
            query=query,
            max_results=max_results,
            date_start=date_start,
            date_end=date_end,
            sort_by=arxiv.SortCriterion.Relevance
        )
    
    def search_seminal_papers(self, topic: str, max_results: int = 10) -> pd.DataFrame:
        """
        Search for potentially seminal papers on a topic
        
        Args:
            topic: Research topic to find seminal papers for
            max_results: Maximum number of papers to fetch
            
        Returns:
            DataFrame with search results
        """
        # Explicitly search for survey papers and reviews which often cite seminal works
        query = f'all:"{topic}" AND (all:"survey" OR all:"review" OR all:"overview")'
        
        # Sort by submission date to get established papers (older papers tend to be more cited)
        # This is a better proxy than using Relevance which may not correlate with citations
        return self.fetch(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
