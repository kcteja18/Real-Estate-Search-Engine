import streamlit as st # type: ignore
import requests
import io

# --- Configuration ---
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Real Estate AI Engine",
    page_icon="🏠",
    layout="wide"
)

# --- Page Definitions ---

def page_ingest():
    """Page for the /ingest endpoint"""
    st.header(" Bulk Property Ingestion")
    
    st.write("**How to use:**")
    st.write("1. Click the 'Browse files' button below.")
    st.write("2. Select the `Property_list.xlsx` file from your computer.")
    st.write("3. Click the 'Start ETL Ingestion' button.")
    st.info("This will read all properties, analyze floorplans with AI, and save everything to the database. This might take a few minutes.")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("Start ETL Ingestion", type="primary"):
            # Prepare file for 'requests'
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            with st.spinner("Processing file... This may take several minutes. Please wait."):
                try:
                    # Set a long timeout (e.g., 10 minutes) for this long-running task
                    response = requests.post(f"{FASTAPI_URL}/ingest", files=files, timeout=600)
                    
                    if response.status_code == 200:
                        st.success("✅ ETL Pipeline Completed Successfully!")
                        st.subheader("Processing Statistics")
                        st.json(response.json().get("stats", {}))
                    else:
                        st.error(f"Error from API (HTTP {response.status_code}):")
                        st.json(response.json()) # Show error details
                        
                except requests.exceptions.Timeout:
                    st.error("Operation timed out. The ETL process is likely still running in the background.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")

def page_analyzer():
    """Page for the /parse-floorplan endpoint"""
    st.header("🛠️ Single Floorplan Analyzer")

    st.write("**How to use:**")
    st.write("1. Click the 'Browse files' button below to upload a floorplan image.")
    st.write("2. Click the 'Analyze Image' button.")
    st.write("3. The AI will count the rooms and display the results.")

    uploaded_image = st.file_uploader("Upload a floorplan image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_image is not None:
        # --- CHANGE 1: Set image width to 400px ---
        st.image(uploaded_image, caption="Uploaded Floorplan", width=400)
        
        if st.button("Analyze Image", type="primary"):
            # Prepare file for 'requests'
            files = {'file': (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
            
            with st.spinner("🤖 AI is analyzing the image..."):
                try:
                    response = requests.post(f"{FASTAPI_URL}/parse-floorplan", files=files)
                    
                    if response.status_code == 200:
                        st.success("✅ Analysis Complete!")
                        results = response.json().get("parsed_counts", {})
                        
                        st.subheader("Parsed Room Counts")
                        col1, col2, col3 = st.columns(3)
                        col4, col5, _ = st.columns([1, 1, 1]) 
                        
                        col1.metric("Rooms (Bedrooms)", results.get("rooms", 0))
                        col2.metric("Bathrooms", results.get("bathrooms", 0))
                        col3.metric("Kitchens", results.get("kitchens", 0))
                        col4.metric("Halls", results.get("halls", 0))
                        col5.metric("Garages", results.get("garages", 0))
                        
                    else:
                        st.error(f"Error from API (HTTP {response.status_code}):")
                        st.json(response.json()) # Show error details
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")

# --- Main App & Sidebar ---

st.title("🏠 Real Estate AI Engine")

# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a tool", ["Data Ingestion", "Floorplan Analyzer"])
st.sidebar.divider()

# --- API Health Check ---
st.sidebar.subheader("API Status")
try:
    response = requests.get(f"{FASTAPI_URL}/")
    if response.status_code == 200:
        data = response.json()
        st.sidebar.success("Backend API: Online")
        if data.get("model_loaded"):
            st.sidebar.info("🤖 AI Model: Loaded")
        else:
            st.sidebar.error("🤖 AI Model: FAILED to Load")
            st.sidebar.warning("Check backend terminal for errors.")
    else:
        st.sidebar.error(f"Backend API: Error (HTTP {response.status_code})")
except requests.ConnectionError:
    st.sidebar.error("Backend API: Offline")
    st.sidebar.warning("Please start the FastAPI server.")

# --- Page Routing ---
if page == "Data Ingestion":
    page_ingest()
elif page == "Floorplan Analyzer":
    page_analyzer()