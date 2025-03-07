import sqlite3
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DBConnector:
    """
    Database connector for managing connections to research papers and citations databases.
    Provides centralized connection management and path resolution.
    """

    
    def __init__(self, research_db_path: str = "backend/database/research_papers.db", 
                 citations_db_path: str = "backend/database/citations.db"):
        # ...rest of the code with proper indentation

        """
        Initialize the database connector with paths to the databases.
        
        Args:
            research_db_path: Path to the research papers database
            citations_db_path: Path to the citations database
        """

# In DBConnector.__init__
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.research_db_path = os.path.join(base_dir, "database", "research_papers.db")
        self.citations_db_path = os.path.join(base_dir, "database", "citations.db")

        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(research_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(citations_db_path), exist_ok=True)
        
        # Store connections
        self._research_conn = None
        self._citations_conn = None
        
        logger.info(f"DBConnector initialized with research_db: {research_db_path}, citations_db: {citations_db_path}")
    
    def get_research_papers_connection(self) -> sqlite3.Connection:
        """Get a connection to the research papers database"""
        if self._research_conn is None or not self._is_connection_valid(self._research_conn):
            try:
                self._research_conn = sqlite3.connect(self.research_db_path)
                logger.info(f"Connected to research papers database at {self.research_db_path}")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to research papers database: {e}")
                raise
        return self._research_conn
    
    def get_citations_connection(self) -> sqlite3.Connection:
        """Get a connection to the citations database"""
        if self._citations_conn is None or not self._is_connection_valid(self._citations_conn):
            try:
                self._citations_conn = sqlite3.connect(self.citations_db_path)
                logger.info(f"Connected to citations database at {self.citations_db_path}")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to citations database: {e}")
                raise
        return self._citations_conn
    
    def _is_connection_valid(self, conn: Optional[sqlite3.Connection]) -> bool:
        """Check if a connection is valid and open"""
        if conn is None:
            return False
        try:
            # Execute a simple query to test the connection
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False
    
    def close_connections(self):
        """Close all database connections"""
        if self._research_conn:
            self._research_conn.close()
            self._research_conn = None
            logger.info("Research papers database connection closed")
        
        if self._citations_conn:
            self._citations_conn.close()
            self._citations_conn = None
            logger.info("Citations database connection closed")
    
    def __del__(self):
        """Destructor to ensure connections are closed when object is destroyed"""
        self.close_connections()
