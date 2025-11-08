import os
import shutil
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any
import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException # type: ignore
from dotenv import load_dotenv

# --- Project Imports ---
# Import functions from your provided files
from inference import RoomDetectionCNN, parse_floorplan
from ETL_pipeline import PropertyETL

# --- Application Setup ---

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a directory for temporary file uploads
TEMP_UPLOAD_DIR = "temp_uploads"
Path(TEMP_UPLOAD_DIR).mkdir(exist_ok=True)


# --- Model Loading (Lifespan Event) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    The model is loaded on startup and stored in app.state.
    """
    logger.info("Application startup...")
    logger.info("Loading model...")
    
    # Get model path from environment or use default
    model_path = os.getenv('MODEL_CHECKPOINT_PATH', 'checkpoint_epoch_50.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found at: {model_path}")
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # Assuming max_rooms_per_type=10 as seen in your ETL_pipeline.py
        model = RoomDetectionCNN(max_rooms_per_type=10)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Store model and device in app state for access in endpoints
        app.state.model = model
        app.state.device = device
        logger.info(f" Model loaded successfully. Using device: {device.upper()}")
        
    except Exception as e:
        logger.critical(f"FATAL: Failed to load AI model: {e}")
        app.state.model = None
        app.state.device = None

    yield  # API is now running

    # --- Shutdown ---
    logger.info("Application shutdown...")
    # You can add cleanup code here if needed
    

# --- FastAPI App Initialization ---

app = FastAPI(
    title="Real Estate Search Engine API",
    description="API for ingesting property data and parsing floorplans.",
    version="1.0.0",
    lifespan=lifespan  # Use the lifespan manager
)


# --- API Endpoints ---

@app.get("/", summary="Health Check")
async def root():
    """
    Root endpoint for health check.
    """
    model_loaded = app.state.model is not None
    return {
        "status": "ok",
        "message": "Real Estate API is running.",
        "model_loaded": model_loaded
    }


@app.post("/ingest", summary="Ingest Property Excel File")
async def ingest_properties(file: UploadFile = File(..., description="The Property_list.xlsx file to ingest.")):
    """
    Uploads an Excel file, saves it temporarily, and triggers the
    full ETL pipeline (as defined in `ETL_pipeline.py`).
    
    This is a long-running operation.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: AI Model is not loaded.")

    # Save the uploaded file to the temporary directory
    temp_path = Path(TEMP_UPLOAD_DIR) / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{file.filename}' saved to '{temp_path}'")

        # --- Run Blocking ETL in a separate thread ---
        def etl_task_sync(file_path: str) -> Dict[str, Any]:
            """Wrapper function to run the synchronous ETL process."""
            logger.info(f"Starting ETL for {file_path}...")
            # Instantiate ETL class pointing to the new file
            etl_pipeline = PropertyETL(excel_path=file_path)
            etl_pipeline.run_etl()
            logger.info(f"ETL completed for {file_path}")
            return etl_pipeline.stats

        # Use asyncio.to_thread to avoid blocking the main server loop
        etl_stats = await asyncio.to_thread(etl_task_sync, str(temp_path))
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": "ETL pipeline completed successfully.",
            "stats": etl_stats
        }

    except Exception as e:
        logger.error(f"ETL pipeline failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"ETL pipeline failed: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if temp_path.exists():
            temp_path.unlink()
            logger.info(f"Temporary file '{temp_path}' deleted.")


@app.post("/parse-floorplan", summary="Parse a Single Floorplan Image")
async def parse_single_floorplan(file: UploadFile = File(..., description="A single floorplan image (e.g., .png, .jpg).")):
    """
    Debug endpoint to upload a single floorplan image and get the model's room count prediction (using `inference.py`).
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Model is not loaded.")

    # Save the uploaded image to the temporary directory
    temp_path = Path(TEMP_UPLOAD_DIR) / file.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # --- Run Blocking Inference in a separate thread ---
        model = app.state.model
        device = app.state.device
        
        # Use asyncio.to_thread for the 'parse_floorplan' function
        prediction = await asyncio.to_thread(parse_floorplan, model, temp_path, device)
        
        if prediction is None:
            raise HTTPException(status_code=422, detail="Failed to process image. It may be corrupted or an invalid format.")
            
        return {
            "filename": file.filename,
            "parsed_counts": prediction
        }

    except Exception as e:
        logger.error(f"Floorplan parsing failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

    finally:
        # Clean up the temporary image file
        if temp_path.exists():
            temp_path.unlink()