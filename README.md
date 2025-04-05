# Research Paper Recommender System

## Overview

This project provides a system for searching, discovering, and evaluating research papers primarily from ArXiv. It features:

* A **FastAPI backend** providing an API for paper search, recommendation, and analysis.
* A **Streamlit frontend** for user interaction with the backend API.
* **ArXiv Integration** to fetch paper metadata.
* **Text Embeddings** using SentenceTransformers and **FAISS** for efficient similarity search.
* **Citation Fetching** (with fallback simulation) to gather citation metrics.
* **Paper Quality Assessment** based on citations, recency, venue, authors, and content.
* **Recommendation Engine** providing content-based recommendations (by text or paper ID) and identification of seminal papers.
* **MongoDB Integration** for storing paper metadata, citation info, and potentially author data.

## Features

* **Search Papers:** Find ArXiv papers using keywords, date ranges, and category filters.
* **Get Recommendations:**
    * Find papers similar to a given abstract or text description.
    * Find papers similar to a specific ArXiv paper ID.
* **Find Seminal Papers:** Identify potentially influential papers based on a research topic.
* **Paper Details:** View title, authors, publication date, abstract, PDF link (if available), categories, similarity score (for recommendations), and a calculated quality score.

## Project Structure

backend/
│
├── app.py # Streamlit frontend application
├── main.py # FastAPI backend application
├── requirements.txt # Python dependencies
├── research_index/ # Stores indexed paper list (indexed_papers.txt)
│ └── indexed_papers.txt
├── database/ # Database related modules
│ ├── __init__.py
│ └── mongo_connector.py # MongoDB connection handler
├── models/ # Core logic modules
│ ├── __init__.py
│ ├── arxiv_fetcher.py # Fetches data from ArXiv
│ ├── citations_fetcher.py # Fetches/simulates citation data
│ ├── embedding_system.py # Handles text embeddings and FAISS index
│ ├── paper_quality_assessor.py # Calculates paper quality score
│ ├── research_recommender.py # Main recommender logic orchestrator
│ └── text_preprocessor.py # Text cleaning and preprocessing
├── logs/ # Directory for log files (created automatically)
└── ... # Other potential configuration files

## Installation

1.  **Clone the repository:**
    `git clone <your-repo-url>`
    `cd backend`

2.  **Create and activate a virtual environment:**
    `python -m venv venv`
    `source venv/bin/activate` # On Windows use `venv\Scripts\activate`

3.  **Install dependencies:**
    `pip install -r requirements.txt`

4.  **(Optional) Configure MongoDB:**
    * The application uses MongoDB for data storage.
    * It defaults to a MongoDB Atlas connection string found in `database/mongo_connector.py`.
    * For production or different setups, set the `MONGODB_URI` environment variable:
        `export MONGODB_URI="your_mongodb_connection_string"`

5.  **Download NLTK data (needed for TextPreprocessor):**
    Run python interpreter: `python`
    Inside python:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    exit()
    ```

## Usage

1.  **Run the FastAPI Backend:**
    `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
    The API will be available at `http://localhost:8000`. Documentation is auto-generated at `http://localhost:8000/docs`.

2.  **Run the Streamlit Frontend:**
    `streamlit run app.py`
    The frontend application will typically open automatically in your browser, usually at `http://localhost:8501`. It connects to the FastAPI backend at the URL specified in `app.py` (default `http://localhost:8000`).
