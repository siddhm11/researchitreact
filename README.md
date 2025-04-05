1. Overview 
The Research Paper Recommender is a web application designed to help users discover relevant 
research papers from arXiv. It consists of a FastAPI backend providing an API for searching, 
recommending, and finding seminal papers, and a Streamlit frontend for user interaction. The 
system fetches data from arXiv and Semantic Scholar, processes paper content using NLP 
techniques, calculates embeddings, assesses paper quality based on citations and other metrics, 
and stores information in a MongoDB database. 
2. Architecture 
The application follows a client-server architecture: 
• Frontend (app.py): A Streamlit application providing the user interface. It interacts with the 
backend API to fetch and display paper information. 
• Backend API (main.py): A FastAPI application exposing endpoints for various 
functionalities. It orchestrates the different components of the system. 
• Core Logic (models/): 
o research_recommender.py: The central class integrating different modules. 
o arxiv_fetcher.py: Fetches paper metadata from the arXiv API. 
o citations_fetcher.py: Fetches citation data, likely using the Semantic Scholar API. 
o embedding_system.py: Generates text embeddings using Sentence Transformers 
and manages a FAISS index for similarity search. 
o text_preprocessor.py: Cleans and preprocesses text data (titles, abstracts). 
o paper_quality_assessor.py: Calculates a quality score for papers based on 
citations, venue, author impact, etc.. 
• Database (database/mongo_connector.py): Handles connection and interaction with a 
MongoDB database to store paper metadata, citation information, etc.. 
3. Setup and Installation 
1. Prerequisites: 
o Python 3.8+ 
o MongoDB instance (Cloud or Local) - Connection URI needed. 
o nltk data (run python -m nltk.downloader punkt stopwords wordnet 
averaged_perceptron_tagger after installing requirements). 
2. Clone Repository: (Assuming code is in a Git repository)  
Bash 
git clone <repository-url> 
cd backend 
3. Create Virtual Environment: 
python -m venv venv 
source venv/bin/activate # On Windows: venv\Scripts\activate 
4. **Install Dependencies:**bash1 
pip install -r requirements.txt2  
``` 
5. Configure Environment: 
* Set the MONGODB_URI environment variable or update the connection string directly in 
database/mongo_connector.py. 
* (Recommended) Update API_BASE_URL in app.py if the frontend and backend run on different 
addresses/ports. 
4. Modules Deep Dive 
• main.py: 
o Sets up the FastAPI application, including lifespan management for 
initializing/shutting down components (like the ResearchRecommender). 
o Defines API request/response models using Pydantic. 
o Implements API endpoints:  
▪ /health: Checks API status. 
▪ /search: Searches papers on arXiv via ResearchRecommender. 
▪ /recommend: Gets paper recommendations based on text or paper ID via 
ResearchRecommender. 
▪ /seminal-papers: Finds seminal papers for a topic via 
ResearchRecommender. 
o Includes CORS middleware and request timing middleware. 
• app.py: 
o Builds the Streamlit UI with pages for "Search Papers", "Get Recommendations", 
and "Find Seminal Papers". 
o Uses the requests library to call the FastAPI backend endpoints. 
o Displays paper details in formatted cards. 
o Includes an API health check indicator. 
• models/research_recommender.py: 
o Orchestrates fetching (ArxivFetcher), embedding (EmbeddingSystem), quality 
assessment (PaperQualityAssessor), and storage (MongoConnector). 
o Provides core methods like search_papers, recommend, find_seminal_papers. 
o Handles loading/saving the FAISS index. 
• models/embedding_system.py: 
o Uses sentence-transformers to load a pre-trained model (e.g., all-MiniLM-L6-v2). 
o Uses faiss-cpu for creating and managing efficient vector similarity indices (both flat 
and IVF). 
o Preprocesses text before embedding using TextPreprocessor. 
o Provides methods for adding papers to the index (process_papers) and finding 
similar papers (find_similar_papers). 
• models/arxiv_fetcher.py: 
o Uses the arxiv library to search and retrieve paper data. 
o Includes retry logic (tenacity) for API calls. 
o Stores fetched papers in MongoDB via MongoConnector. 
• models/citations_fetcher.py: 
o Designed to fetch citation counts, author details (h-index), etc., likely from 
Semantic Scholar API. 
o Includes retry logic. 
o Stores citation and author data in MongoDB. 
• models/paper_quality_assessor.py: 
o Calculates a composite quality score based on citation counts, influential citations, 
author impact (h-index), venue prestige (uses a simple tier list), and paper recency. 
o Relies on CitationsFetcher to get necessary data. 
• models/text_preprocessor.py: 
o Provides text cleaning functions (removing URLs, special characters, LaTeX, 
citations) and normalization (lowercasing, stopword removal, lemmatization) using 
nltk. 
• database/mongo_connector.py: 
o Manages the connection to MongoDB using pymongo. 
o Provides access to different collections (papers, citations, authors). 
5. API Endpoints (main.py) 
• GET /health: Returns the status of the API components. 
• POST /search:  
o Request Body: SearchRequest model (query, max_results, optional date_range, 
optional categories). 
o Response: List of paper objects matching the search criteria. 
• POST /recommend:  
o Request Body: RecommendRequest model (optional text, optional paper_id, k 
recommendations, optional date_range, quality_aware flag). 
o Response: List of recommended paper objects with similarity scores. 
• POST /seminal-papers:  
o Request Body: SeminalPapersRequest model (topic, max_results). 
o Response: List of potentially seminal paper objects related to the topic.
