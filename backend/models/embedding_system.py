import numpy as np
import pandas as pd
import faiss
import logging
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

# You'll need to import the TextPreprocessor class that you'll create
from .text_preprocessor import TextPreprocessor

# Set up logger
logger = logging.getLogger(__name__)

class EmbeddingSystem:
    """
    Embedding System to encode research papers and perform similarity search.
    Uses FAISS for fast indexing and optimized vector representations.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        # Create two indices:
        # 1. A flat index for accurate reconstruction
        self.flat_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 2. IVF index for faster search with minimal accuracy loss
        self.quantizer = faiss.IndexFlatIP(self.embedding_dim)
        # Use 4x sqrt(n) clusters for better balance of speed and accuracy
        # We'll initialize with 100 clusters and retrain as needed
        self.index = faiss.IndexIVFFlat(self.quantizer, self.embedding_dim, 100, faiss.METRIC_INNER_PRODUCT)
        self.index_trained = False
        
        # Store embeddings and their mapping to papers
        self.metadata = pd.DataFrame()
        # Store raw embeddings for direct lookup
        self.raw_embeddings = {}  # Map paper_id to embedding vector
        
        self.preprocessor = TextPreprocessor()
    
    def limit_embedding_cache(self, max_size: int = 10000):
        """
        Limit the size of the raw_embeddings cache to prevent memory issues
        
        Args:
            max_size: Maximum number of embeddings to keep in cache
        """
        if len(self.raw_embeddings) > max_size:
            logger.info(f"Trimming embedding cache from {len(self.raw_embeddings)} to {max_size} entries")
            # Keep only the most recent max_size entries
            # Convert to list of (key, value) tuples, sort by keys or values if needed
            items = list(self.raw_embeddings.items())
            # For simplicity, we'll just keep the last max_size items
            # In practice, you might want a more sophisticated strategy
            self.raw_embeddings = dict(items[-max_size:])
        
    def generate_embeddings(self, texts: list) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Process in batches for memory efficiency
        batch_size = 64  # Increased from 32 for MiniLM
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False).astype('float32')
            all_embeddings.append(batch_embeddings)
            
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Embeddings generated: {embeddings.shape}")
        return embeddings
    
    def prepare_text_for_embedding(self, title: str, abstract: str) -> str:
        """Ensure title and abstract are processed even if missing."""
        clean_title = self.preprocessor.clean_text(title) if title else "Untitled Paper"
        clean_abstract = self.preprocessor.clean_text(abstract) if abstract else "No abstract available"
        return f"{clean_title} [SEP] {clean_title} [SEP] {clean_abstract}"

    
    def process_papers(self, df: pd.DataFrame, preprocess: bool = True) -> None:
        if df.empty:
            logger.warning("⚠️ No papers to process, skipping FAISS indexing.")
            return

        if "title" not in df.columns or "abstract" not in df.columns:
            logger.error("❌ Missing required columns ('title' or 'abstract') in DataFrame.")
            return

        # Ensure clean_text column exists
        df['clean_text'] = df.apply(lambda row: self.prepare_text_for_embedding(row['title'], row.get('abstract', "")), axis=1)

        if df['clean_text'].isnull().all():
            logger.error("❌ All clean_text values are empty! Stopping FAISS processing.")
            return

        logger.info(f"✅ Processing {len(df)} papers into FAISS index.")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df['clean_text'].tolist())

        # Train IVF index if necessary
        if not self.index_trained or self.index.ntotal < 1000:
            n_clusters = min(4 * int(np.sqrt(len(df) + self.index.ntotal)), 256)
            n_clusters = max(n_clusters, 100)

            logger.info(f"Training IVF index with {n_clusters} clusters...")

            self.index = faiss.IndexIVFFlat(self.quantizer, self.embedding_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)

            if len(embeddings) < n_clusters:
                logger.warning(f"Not enough vectors to train {n_clusters} clusters. Using simple FAISS index.")
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Fallback to Flat index
                self.index_trained = True
            else:
                self.index.train(embeddings)
                self.index_trained = True

        # Add embeddings to FAISS index
        if self.index_trained:
            self.index.add(embeddings)
            logger.info(f"📌 FAISS index now contains {self.index.ntotal} embeddings.")
        else:
            logger.warning("Index not trained. Cannot add vectors.")
            return

        # Store metadata
        paper_df = df.copy()
        current_size = len(self.metadata)
        paper_df['embedding_idx'] = list(range(current_size, current_size + len(df)))

        if self.metadata.empty:
            self.metadata = paper_df
        else:
            self.metadata = pd.concat([self.metadata, paper_df], ignore_index=True)

        # Preprocess text if required
        if preprocess:
            logger.info("Adding preprocessed text for improved recommendations...")
            processed_text_map = {}

            if len(df) > 50:
                processed_texts = self.preprocessor.batch_process(df['clean_text'].tolist(), lemmatize=True)
                for i, (_, row) in enumerate(df.iterrows()):
                    processed_text_map[row['paper_id']] = processed_texts[i]
            else:
                for _, row in df.iterrows():
                    processed_text = self.preprocessor.process_text(row['clean_text'], lemmatize=True)
                    processed_text_map[row['paper_id']] = processed_text

            # Apply processed text mapping
            self.metadata.loc[self.metadata['paper_id'].isin(processed_text_map.keys()), 'processed_text'] = \
                self.metadata.loc[self.metadata['paper_id'].isin(processed_text_map.keys()), 'paper_id'].map(processed_text_map)

        logger.info("✅ Metadata stored with embeddings.")

    def get_paper_embedding(self, paper_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a specific paper ID
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Numpy array containing the embedding vector or None if not found
        """
        # First try direct lookup from stored raw embeddings (fastest)
        if paper_id in self.raw_embeddings:
            return self.raw_embeddings[paper_id].reshape(1, -1)
        
        # Fallback to metadata lookup and flat index reconstruction
        paper_data = self.metadata[self.metadata['paper_id'] == paper_id]
        if paper_data.empty:
            logger.warning(f"Paper ID {paper_id} not found in index.")
            return None
        
        # Get embedding index for this paper
        if 'embedding_idx' not in paper_data.columns:
            logger.warning(f"No embedding index for paper ID {paper_id}")
            return None
            
        embedding_idx = int(paper_data['embedding_idx'].iloc[0])
        
        # Reconstruct embedding from flat index (reliable)
        try:
            vector = np.zeros((1, self.embedding_dim), dtype='float32')
            vector[0] = self.flat_index.reconstruct(embedding_idx)
            return vector
        except Exception as e:
            logger.error(f"Error reconstructing embedding for paper ID {paper_id}: {e}")
            return None
    
    def get_fallback_recommendations(self, count: int = 5) -> pd.DataFrame:
        """
        Get fallback recommendations when not enough results are found
        
        Args:
            count: Number of fallback recommendations to return
            
        Returns:
            DataFrame with fallback paper recommendations
        """
        # This is a stub - you'll need to implement this method
        # It should return the most popular or most recent papers
        logger.info(f"Getting {count} fallback recommendations")
        
        if self.metadata.empty or len(self.metadata) <= count:
            return self.metadata
            
        # You could implement various strategies here:
        # 1. Return the most recent papers
        # 2. Return papers with highest citation counts
        # 3. Return papers from top venues/journals
        # For now, let's just return random papers
        
        return self.metadata.sample(min(count, len(self.metadata)))

    def recommend(self, 
              text: str = None, 
              paper_id: str = None,
              user_preferences: np.ndarray = None,
              k: int = 5, 
              filter_criteria: Dict = None,
              nprobe: int = 10,
              quality_assessor: Optional['PaperQualityAssessor'] = None) -> pd.DataFrame:
        """
        Get recommendations based on text, paper_id, or user preferences
        
        Args:
            text: Input text to find similar papers
            paper_id: ID of paper to find similar papers to
            user_preferences: Pre-computed user preference vector
            k: Number of recommendations to return
            filter_criteria: Dictionary of metadata filters to apply
            nprobe: Number of clusters to probe in IVF index (higher = more accurate but slower)
            quality_assessor: PaperQualityAssessor instance for quality scoring
            
        Returns:
            DataFrame with recommended papers and similarity scores
        """
        if not self.index_trained or self.index.ntotal == 0:
            logger.warning("Index not trained or empty. Cannot recommend papers.")
            return pd.DataFrame()

        if hasattr(self.index, 'nprobe'):  # Set nprobe only if using IVF index
            self.index.nprobe = nprobe

        query_vector = None
        matched_titles = []
        
        # 🔹 Case 1: Text query
        if text:
            logger.info(f"Generating embedding for query text: {text[:50]}...")
            clean_text = self.preprocessor.clean_text(text).lower()

            # 🔹 Perform fuzzy matching with titles
            matched_titles = [
                paper for paper in self.metadata.to_dict(orient="records")
                if fuzz.ratio(clean_text, paper["title"].lower()) >= 60  # Adjust threshold as needed
            ]

            # Sort fuzzy matches by highest similarity
            matched_titles.sort(key=lambda x: fuzz.ratio(clean_text, x["title"].lower()), reverse=True)

            if matched_titles:
                logger.info(f"Found {len(matched_titles)} fuzzy matches for '{text}'")

            # 🔹 Encode query text into embedding vector
            query_vector = self.model.encode([clean_text])[0].astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vector)

        # 🔹 Case 2: Paper ID query
        elif paper_id:
            logger.info(f"Finding papers similar to paper_id: {paper_id}")
            query_vector = self.get_paper_embedding(paper_id)
            
            if query_vector is None:
                logger.warning(f"Could not retrieve embedding for paper ID {paper_id}")
                return pd.DataFrame()

        # 🔹 Case 3: User Preferences
        elif user_preferences is not None:
            logger.info("Using provided user preference vector for recommendations")
            query_vector = user_preferences.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_vector)

        else:
            logger.error("No query provided. Please provide text, paper_id, or user_preferences.")
            return pd.DataFrame()
        
        # 🔹 Perform FAISS similarity search
        num_results = min(2 * k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, num_results)

        # 🔹 Convert FAISS results into DataFrame
        results_df = pd.DataFrame({
            'embedding_idx': indices[0],
            'similarity_score': scores[0]
        })

        # 🔹 Merge FAISS results with metadata
        results_with_metadata = []
        for _, row in results_df.iterrows():
            embedding_idx = int(row['embedding_idx'])
            metadata_matches = self.metadata[self.metadata['embedding_idx'] == embedding_idx]

            if not metadata_matches.empty:
                paper_data = metadata_matches.iloc[0].to_dict()
                paper_data['similarity_score'] = row['similarity_score']
                results_with_metadata.append(paper_data)

        if not results_with_metadata:
            logger.warning("No metadata matches found for search results.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results_with_metadata)

        # 🔹 Merge fuzzy title matches **with FAISS results**, removing duplicates
        if matched_titles:
            fuzzy_df = pd.DataFrame(matched_titles)
            results_df = pd.concat([fuzzy_df, results_df]).drop_duplicates(subset=['paper_id']).reset_index(drop=True)

        # 🔹 Ensure at least `k` results by using fallback recommendations
        if len(results_df) < k:
            logger.warning("Not enough results, fetching fallback recommendations.")
            fallback_papers = self.get_fallback_recommendations(k - len(results_df))
            results_df = pd.concat([results_df, fallback_papers]).reset_index(drop=True)

        # 🔹 Remove the query paper from results if searching by paper_id
        if paper_id:
            results_df = results_df[results_df['paper_id'] != paper_id]

        # 🔹 Apply filters if provided
        if filter_criteria:
            for column, value in filter_criteria.items():
                if column in results_df.columns:
                    if isinstance(value, list):
                        results_df = results_df[results_df[column].isin(value)]
                    elif isinstance(value, dict) and 'min' in value and 'max' in value:
                        results_df = results_df[(results_df[column] >= value['min']) & (results_df[column] <= value['max'])]
                    elif isinstance(value, dict) and 'after' in value:
                        results_df = results_df[results_df[column] >= value['after']]
                    elif isinstance(value, dict) and 'before' in value:
                        results_df = results_df[results_df[column] <= value['before']]
                    else:
                        results_df = results_df[results_df[column] == value]

        # 🔹 Apply quality assessment if enabled
        if quality_assessor and not results_df.empty:
            logger.info("Applying quality assessment to recommendations...")
            results_df['quality_score'] = results_df.apply(lambda paper: quality_assessor.assess_paper_quality(paper), axis=1)
            
            similarity_weight = 0.7  # 70% weight for similarity
            quality_weight = 0.3  # 30% weight for quality
            
            results_df['combined_score'] = (similarity_weight * results_df['similarity_score']) + (quality_weight * results_df['quality_score'])
            results_df = results_df.sort_values('combined_score', ascending=False)
        else:
            results_df = results_df.sort_values('similarity_score', ascending=False)

        # 🔹 Normalize similarity score to percentage
        results_df['similarity_percent'] = (results_df['similarity_score'] * 100).round(2)

        return results_df.head(k)