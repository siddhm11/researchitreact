import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import time

# Configure the app
st.set_page_config(
    page_title="Research Paper Recommender",
    page_icon="📚",
    layout="wide"
)

# API base URL - change this to your FastAPI server URL
API_BASE_URL = "http://localhost:8000"

# Main title
st.title("📚 Research Paper Recommender")
st.markdown("Find and discover relevant research papers based on your interests.")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Search Papers", "Get Recommendations", "Find Seminal Papers"])

# Function to format the paper card
def paper_card(paper):
    with st.container():
        st.markdown(f"### {paper['title']}")
        st.markdown(f"**Authors**: {', '.join(paper['authors'])}")
        st.markdown(f"**Published**: {paper['published']}")
        st.markdown(f"**Categories**: {', '.join(paper['categories']) if paper['categories'] else 'N/A'}")
        
        if 'similarity' in paper and paper['similarity'] is not None:
            st.progress(float(paper['similarity'])/100)
            st.markdown(f"**Similarity Score**: {paper['similarity']:.2f}%")
            
        if 'quality_score' in paper and paper['quality_score'] is not None:
            quality_color = "green" if paper['quality_score'] > 0.7 else "orange" if paper['quality_score'] > 0.4 else "red"
            st.markdown(f"**Quality Score**: <span style='color:{quality_color}'>{paper['quality_score']:.2f}</span>", unsafe_allow_html=True)
        
        with st.expander("Abstract"):
            st.markdown(paper['abstract'])
            
        col1, col2 = st.columns([1, 6])
        if paper.get('pdf_url'):
            with col1:
                st.markdown(f"[PDF Link]({paper['pdf_url']})")
                
        with col2:
            if st.button(f"Get Similar Papers", key=f"sim_{paper['paper_id']}"):
                # Set session state to navigate to recommendations with this paper ID
                st.session_state.recommend_paper_id = paper['paper_id']
                st.session_state.page = "Get Recommendations"
                st.rerun()
                
        st.markdown("---")

# Function to check API health
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, {"status": "API responded with error"}
    except requests.exceptions.RequestException:
        return False, {"status": "API not reachable"}

# Check API health in sidebar
api_status, health_info = check_api_health()
if api_status:
    st.sidebar.success("✅ API Connected")
    st.sidebar.json(health_info)
else:
    st.sidebar.error("❌ API Not Connected")
    st.sidebar.warning("Make sure your FastAPI backend is running")
    st.sidebar.json(health_info)

# Search Papers page
if page == "Search Papers":
    st.header("Search for Research Papers")
    
    with st.form("search_form"):
        query = st.text_input("Search Query", placeholder="e.g. transformer nlp")
        
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("Maximum Results", min_value=5, max_value=100, value=20)
        with col2:
            use_date_filter = st.checkbox("Filter by Date")
        
        if use_date_filter:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", datetime.now().replace(year=datetime.now().year-1))
            with date_col2:
                end_date = st.date_input("End Date", datetime.now())
        
        categories = st.multiselect(
            "Filter by Categories",
            ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML", "cs.IR", "cs.RO", "cs.HC"]
        )
        
        search_button = st.form_submit_button("Search Papers")
    
    if search_button and query:
        with st.spinner("Searching for papers..."):
            # Prepare request data
            req_data = {
                "query": query,
                "max_results": max_results
            }
            
            if use_date_filter:
                req_data["date_range"] = {
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d")
                }
            
            if categories:
                req_data["categories"] = categories
            
            try:
                response = requests.post(f"{API_BASE_URL}/search", json=req_data, timeout=60)
                
                if response.status_code == 200:
                    papers = response.json()
                    if papers:
                        st.success(f"Found {len(papers)} relevant papers")
                        for paper in papers:
                            paper_card(paper)
                    else:
                        st.info("No papers found matching your criteria.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {str(e)}")

# Get Recommendations page
elif page == "Get Recommendations":
    st.header("Get Paper Recommendations")
    
    # Initialize session state for recommendation method
    if 'recommend_method' not in st.session_state:
        st.session_state.recommend_method = 'text'
    
    # Initialize paper_id if coming from another page
    paper_id_input = ""
    if 'recommend_paper_id' in st.session_state:
        paper_id_input = st.session_state.recommend_paper_id
        st.session_state.recommend_method = 'paper_id'
        # Clear after using
        del st.session_state.recommend_paper_id
    
    # Method selector
    rec_method = st.radio(
        "Recommendation Method",
        ["Based on Text", "Based on Paper ID"],
        horizontal=True,
        index=0 if st.session_state.recommend_method == 'text' else 1
    )
    
    st.session_state.recommend_method = 'text' if rec_method == "Based on Text" else 'paper_id'
    
    with st.form("recommend_form"):
        if st.session_state.recommend_method == 'text':
            input_text = st.text_area(
                "Enter Text to Find Similar Papers",
                placeholder="Paste abstract, research proposal, or description here...",
                height=150
            )
            input_source = input_text
            paper_id = None
        else:
            paper_id = st.text_input(
                "Paper ID (arXiv ID)",
                value=paper_id_input,
                placeholder="e.g. 2103.12345"
            )
            input_source = paper_id
            input_text = None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            k = st.slider("Number of Recommendations", min_value=1, max_value=50, value=5)
        
        with col2:
            use_date_filter = st.checkbox("Filter by Date")
        
        with col3:
            quality_aware = st.checkbox("Use Quality Assessment", value=True)
        
        if use_date_filter:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", datetime.now().replace(year=datetime.now().year-2))
            with date_col2:
                end_date = st.date_input("End Date", datetime.now())
        
        recommend_button = st.form_submit_button("Get Recommendations")
    
    if recommend_button:
        if (st.session_state.recommend_method == 'text' and input_text) or (st.session_state.recommend_method == 'paper_id' and paper_id):
            with st.spinner("Finding recommendations..."):
                # Prepare request data
                req_data = {
                    "k": k,
                    "quality_aware": quality_aware
                }
                
                if st.session_state.recommend_method == 'text':
                    req_data["text"] = input_text
                else:
                    req_data["paper_id"] = paper_id
                
                if use_date_filter:
                    req_data["date_range"] = {
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d")
                    }
                
                try:
                    start_time = time.time()
                    response = requests.post(f"{API_BASE_URL}/recommend", json=req_data, timeout=120)
                    elapsed_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        papers = response.json()
                        if papers:
                            st.success(f"Found {len(papers)} recommendations in {elapsed_time:.2f} seconds")
                            for paper in papers:
                                paper_card(paper)
                        else:
                            st.info("No recommendations found. Try different input or parameters.")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {str(e)}")
        else:
            st.warning("Please provide input for recommendation.")

# Find Seminal Papers page
elif page == "Find Seminal Papers":
    st.header("Find Seminal Papers on a Topic")
    
    with st.form("seminal_form"):
        topic = st.text_input("Research Topic", placeholder="e.g. attention mechanisms")
        max_results = st.slider("Maximum Results", min_value=1, max_value=20, value=5)
        
        seminal_button = st.form_submit_button("Find Seminal Papers")
    
    if seminal_button and topic:
        with st.spinner("Searching for seminal papers..."):
            # Prepare request data
            req_data = {
                "topic": topic,
                "max_results": max_results
            }
            
            try:
                response = requests.post(f"{API_BASE_URL}/seminal-papers", json=req_data, timeout=60)
                
                if response.status_code == 200:
                    papers = response.json()
                    if papers:
                        st.success(f"Found {len(papers)} seminal papers on {topic}")
                        for paper in papers:
                            paper_card(paper)
                    else:
                        st.info(f"No seminal papers found for topic: {topic}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("📚 **Research Paper Recommender** | Streamlit + FastAPI Integration")