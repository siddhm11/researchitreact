import sqlite3
import json
import pandas as pd
import datetime
from typing import List, Optional
import arxiv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logger
logger = logging.getLogger(__name__)

class ArxivFetcher:
    """
    Fetches research papers from Arxiv API with enhanced filtering.
    Implements retry mechanism for reliability.
    """
    
    def __init__(self, db_path=None):
        if db_path is None:
            import os
            # Get the directory of the current script
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "database", "research_papers.db")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    
    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            primary_category TEXT,
            categories TEXT,
            published DATE,
            pdf_url TEXT,
            source TEXT,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""  # Add closing parenthesis
        self.conn.execute(query)
        self.conn.commit()
        
    def __del__(self):
        """Close database connection when object is destroyed"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def fetch_store(self, query="cat:cs.LG", max_results=100):
        """Fetch papers and store only new ones in the database."""

        # Step 1: Fetch new papers
        df = self.fetch(query=query, max_results=max_results)
        
        if df.empty:
            print("⚠️ No papers retrieved! Skipping storage.")
            return df  # Return empty DataFrame if nothing fetched

        # Step 2: Get already stored paper IDs
        existing_papers = pd.read_sql("SELECT paper_id FROM papers", self.conn)
        existing_ids = set(existing_papers["paper_id"])

        # Rename 'id' column to 'paper_id' to match database schema
        if 'id' in df.columns and 'paper_id' not in df.columns:
            df = df.rename(columns={'id': 'paper_id'})  # ✅ Ensure correct naming


        # Step 3: Filter out papers that are already stored
        new_df = df[~df["paper_id"].isin(existing_ids)]

        if new_df.empty:
            print("🔄 All fetched papers are already stored. No new papers added.")
        else:
            # Step 4: Store only new papers
            new_df.to_sql("papers", self.conn, if_exists="append", index=False)
            print(f"✅ Stored {len(new_df)} new papers.")

        return new_df

    def check_database(self):
        """retrieve stored papers"""   
        query = "SELECT * FROM papers"
        return pd.read_sql(query, self.conn)
        
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
                return pd.DataFrame()

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