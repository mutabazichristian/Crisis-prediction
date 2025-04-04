from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import os
import shutil
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import sys
import logging
import threading
import time
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add src directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, 'static')
templates_dir = os.path.join(BASE_DIR, 'templates')

# Import local modules
from src.prediction import PredictionService
from src.pipeline import MLPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Banking Crisis Prediction",
    description="ML API for predicting banking crises in African countries",
    version="1.0.0"
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=templates_dir)

# Initialize prediction service
try:
    prediction_service = PredictionService()
    logger.info("Prediction service initialized successfully")
except Exception as e:
    logger.error(f"Error initializing prediction service: {e}")
    prediction_service = None

# Initialize ML pipeline
try:
    ml_pipeline = MLPipeline()
    logger.info("ML pipeline initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ML pipeline: {e}")
    ml_pipeline = None

# Define data models
class PredictionInput(BaseModel):
    country: int
    year: int
    exch_usd: float
    gdp_weighted_default: float
    inflation_annual_cpi: float
    systemic_crisis: int = 0
    
class BatchPredictionInput(BaseModel):
    data: List[Dict[str, Any]]

# Required columns for the dataset
REQUIRED_COLUMNS = [
    'country',
    'year',
    'exch_usd',
    'gdp_weighted_default',
    'inflation_annual_cpi',
    'systemic_crisis',
    'banking_crisis'
]

def validate_csv_structure(df):
    """Validate that DataFrame has all required columns"""
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    return True

# Global dictionary to store training status
training_status = {}

# Add thread synchronization
status_lock = threading.Lock()
training_threads = {}

def update_training_status(filename: str, status: str, step: int = 0, error: str = None, metrics: dict = None):
    """Update training status with thread safety."""
    with status_lock:
        training_status[filename] = {
            "status": status,
            "step": step,
            "error": error,
            "metrics": metrics,
            "timestamp": time.time(),
            "last_update": datetime.now().isoformat()
        }
        logger.info(f"Status updated for {filename}: status={status}, step={step}")

def run_training(filename: str):
    """Run the training pipeline for a given file."""
    try:
        # Set initial status
        update_training_status(filename, "training", step=0)
        
        # Use absolute path for file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "..", "data", "uploads", filename)
        
        if not os.path.exists(file_path):
            raise ValueError(f"Training file not found: {file_path}")
            
        logger.info(f"Starting training with file: {file_path}")
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")
        
        # Step 1: Load and validate data
        update_training_status(filename, "training", step=1)
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read training file. Shape: {df.shape}")
        except Exception as e:
            raise ValueError(f"Failed to read training file: {str(e)}")
        
        # Step 2: Initialize pipeline
        update_training_status(filename, "training", step=2)
        pipeline = MLPipeline()
        
        # Step 3: Start training
        update_training_status(filename, "training", step=3)
        metrics = pipeline.run_training_pipeline(file_path)
        
        if not metrics or metrics.get("status") != "success":
            raise ValueError("Training failed: No metrics returned")
            
        # Step 4: Save model
        update_training_status(filename, "training", step=4)
        
        # Step 5: Complete
        logger.info(f"Training completed successfully. Metrics: {metrics}")
        update_training_status(filename, "completed", step=5, metrics=metrics.get("metrics", {}))
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        update_training_status(filename, "error", error=str(e))
        # Don't re-raise the exception since this is in a background thread

@app.get("/api/training-status")
async def get_training_status(filename: str):
    """Get the current training status."""
    logger.debug(f"Getting status for {filename}. Current status: {training_status.get(filename)}")
    if filename not in training_status:
        return {"status": "not_found"}
    return training_status[filename]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction routes
@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    """Render prediction form"""
    # List of African countries for the prediction form
    countries = [
        "Algeria",
        "Angola",
        "Central African Republic",
        "Egypt",
        "Ivory Coast",
        "Kenya",
        "Mauritius",
        "Morocco",
        "Nigeria",
        "South Africa",
        "Tunisia",
        "Zambia",
        "Zimbabwe"
    ]
    return templates.TemplateResponse(
        "predict.html", 
        {
            "request": request,
            "countries": countries,
            "current_year": 2024  # Add current year as default
        }
    )

@app.post("/predict")
async def predict(
    request: Request,
    country: str = Form(...),
    year: int = Form(...),
    exch_usd: float = Form(...),
    gdp_weighted_default: float = Form(...),
    inflation_annual_cpi: float = Form(...),
    systemic_crisis: int = Form(0)
):
    """Make prediction based on form input"""
    try:
        # List of countries (must match the order in the model training)
        countries = [
            "Algeria",
            "Angola",
            "Central African Republic",
            "Egypt",
            "Ivory Coast",
            "Kenya",
            "Mauritius",
            "Morocco",
            "Nigeria",
            "South Africa",
            "Tunisia",
            "Zambia",
            "Zimbabwe"
        ]
        
        # Convert country name to index
        try:
            country_idx = countries.index(country)
        except ValueError:
            raise ValueError(f"Invalid country: {country}")
        
        # Create input data
        input_data = pd.DataFrame({
            "country": [country_idx],
            "year": [year],
            "exch_usd": [exch_usd],
            "gdp_weighted_default": [gdp_weighted_default],
            "inflation_annual_cpi": [inflation_annual_cpi],
            "systemic_crisis": [systemic_crisis]
        })
        
        # Make prediction
        if prediction_service is None:
            return templates.TemplateResponse(
                "error.html", 
                {"request": request, "message": "Prediction service not available"}
            )
        
        result = prediction_service.predict(input_data.iloc[0].to_dict())
        
        # Add country name to input data for display
        input_data = input_data.iloc[0].to_dict()
        input_data["country"] = country
        
        # Render result page
        return templates.TemplateResponse(
            "prediction_result.html", 
            {"request": request, "result": result, "input_data": input_data}
        )
    except ValueError as e:
        logger.error(f"Validation error in prediction: {e}")
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": str(e)}
        )

@app.post("/api/predict", response_model=Dict[str, Any])
async def predict_api(input_data: PredictionInput):
    """API endpoint for prediction"""
    try:
        # Make prediction
        if prediction_service is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Prediction service not available"}
            )
        
        result = prediction_service.predict(input_data.dict())
        return result
    except Exception as e:
        logger.error(f"Error in prediction API: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/batch-predict")
async def batch_predict_api(input_data: BatchPredictionInput):
    """API endpoint for batch prediction"""
    try:
        # Make batch prediction
        if prediction_service is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Prediction service not available"}
            )
        
        results = prediction_service.batch_predict(input_data.data)
        return {"predictions": results}
    except Exception as e:
        logger.error(f"Error in batch prediction API: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Training routes
@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Render data upload form"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_data(request: Request, file: UploadFile = File(...)):
    """Upload new data file and trigger retraining"""
    try:
        # Verify file extension
        if not file.filename.endswith('.csv'):
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "message": "Invalid file format. Only CSV files are accepted."
                }
            )

        # Create upload directory if it doesn't exist using absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(base_dir, "..", "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}.csv"
        file_path = os.path.join(upload_dir, safe_filename)
        
        try:
            # Read the contents of the uploaded file
            contents = await file.read()
            if not contents:
                raise ValueError("The uploaded file is empty")
            
            logger.info(f"Read file contents, size: {len(contents)} bytes")
            
            # Write contents to disk
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            
            logger.info(f"Wrote file to {file_path}")
            
            # Verify the file exists and has content
            if not os.path.exists(file_path):
                raise ValueError("Failed to write file to disk")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError("Written file is empty")
            
            logger.info(f"File size on disk: {file_size} bytes")
            
            # Try to read the CSV file with different encodings if needed
            df = None
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            last_error = None
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Attempting to read CSV with {encoding} encoding")
                    df = pd.read_csv(file_path, encoding=encoding)
                    if not df.empty:
                        logger.info(f"Successfully read CSV with {encoding} encoding. Shape: {df.shape}")
                        break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Failed to read CSV with {encoding} encoding: {str(e)}")
                    continue
            
            if df is None or df.empty:
                raise ValueError(f"Could not read the CSV file with any supported encoding. Last error: {last_error}")
            
            # Log DataFrame info for debugging
            logger.info(f"DataFrame info:")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Data types:\n{df.dtypes}")
            
            # Validate CSV structure
            validate_csv_structure(df)
            
            # Validate data types and convert them
            numeric_columns = {
                'year': 'int64',
                'exch_usd': 'float64',
                'gdp_weighted_default': 'float64',
                'inflation_annual_cpi': 'float64',
                'systemic_crisis': 'int64'
            }
            
            for col, dtype in numeric_columns.items():
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                    df[col] = df[col].astype(dtype)
                    logger.info(f"Successfully converted {col} to {dtype}")
                except Exception as e:
                    raise ValueError(f"Invalid data in column {col}: {str(e)}")
            
            # Validate value ranges
            if not all(df['systemic_crisis'].isin([0, 1])):
                raise ValueError("systemic_crisis must be binary (0 or 1)")
            
            if not all(df['banking_crisis'].isin(['crisis', 'no_crisis'])):
                raise ValueError("banking_crisis must be 'crisis' or 'no_crisis'")
            
            rows, cols = df.shape
            logger.info(f"File validated for training: {safe_filename}, {rows} rows, {cols} columns")
            
            # Save validated DataFrame back to CSV with UTF-8 encoding
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Saved validated DataFrame back to {file_path}")
            
            # Start training in background
            thread = threading.Thread(target=run_training, args=(safe_filename,))
            thread.daemon = True
            thread.start()
            
            # Update training status
            update_training_status(safe_filename, "started", step=0)
            
            logger.info(f"Training thread started for {safe_filename}")
            
            # Return success page with file info and training status
            return templates.TemplateResponse(
                "upload_success.html",
                {
                    "request": request,
                    "filename": safe_filename,
                    "rows": rows,
                    "cols": cols,
                    "training_started": True
                }
            )
                
        except Exception as e:
            # Clean up the file if validation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error in file processing: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing CSV: {str(e)}")
            
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error during upload and training: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"An unexpected error occurred: {str(e)}"}
        )

@app.post("/retrain")
async def retrain_model(request: Request, file: UploadFile = File(...)):
    """Start model retraining."""
    try:
        # Verify file extension
        if not file.filename.endswith('.csv'):
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "message": "Invalid file format. Only CSV files are accepted."}
            )

        # Create upload directory if it doesn't exist using absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(base_dir, "..", "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}.csv"
        file_path = os.path.join(upload_dir, safe_filename)
        
        try:
            # First, save the uploaded file to disk using SpooledTemporaryFile
            contents = await file.read()
            if not contents:
                raise ValueError("The uploaded file is empty")
            
            logger.info(f"Read file contents, size: {len(contents)} bytes")
            
            # Write contents to disk
            with open(file_path, "wb") as buffer:
                buffer.write(contents)
            
            logger.info(f"Wrote file to {file_path}")
            
            # Verify the file exists and has content
            if not os.path.exists(file_path):
                raise ValueError("Failed to write file to disk")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError("Written file is empty")
                
            logger.info(f"File size on disk: {file_size} bytes")
            
            # Try to read the CSV file with different encodings if needed
            df = None
            encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            last_error = None
            
            for encoding in encodings_to_try:
                try:
                    logger.info(f"Attempting to read CSV with {encoding} encoding")
                    # Read the file directly from disk
                    df = pd.read_csv(file_path, encoding=encoding)
                    if not df.empty:
                        logger.info(f"Successfully read CSV with {encoding} encoding. Shape: {df.shape}")
                        break
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Failed to read CSV with {encoding} encoding: {str(e)}")
                    continue
            
            if df is None or df.empty:
                raise ValueError(f"Could not read the CSV file with any supported encoding. Last error: {last_error}")
            
            # Log DataFrame info for debugging
            logger.info(f"DataFrame info:")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Data types:\n{df.dtypes}")
            
            # Validate CSV structure
            validate_csv_structure(df)
            
            # Validate data types and convert them
            numeric_columns = {
                'year': 'int64',
                'exch_usd': 'float64',
                'gdp_weighted_default': 'float64',
                'inflation_annual_cpi': 'float64',
                'systemic_crisis': 'int64'
            }
            
            for col, dtype in numeric_columns.items():
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                    df[col] = df[col].astype(dtype)
                    logger.info(f"Successfully converted {col} to {dtype}")
                except Exception as e:
                    raise ValueError(f"Invalid data in column {col}: {str(e)}")
            
            # Validate value ranges
            if not all(df['systemic_crisis'].isin([0, 1])):
                raise ValueError("systemic_crisis must be binary (0 or 1)")
            
            if not all(df['banking_crisis'].isin(['crisis', 'no_crisis'])):
                raise ValueError("banking_crisis must be 'crisis' or 'no_crisis'")
            
            rows, cols = df.shape
            logger.info(f"File validated for training: {safe_filename}, {rows} rows, {cols} columns")
            
            # Save validated DataFrame back to CSV with UTF-8 encoding
            df.to_csv(file_path, index=False, encoding='utf-8')
            logger.info(f"Saved validated DataFrame back to {file_path}")
            
            # Start training in background
            thread = threading.Thread(target=run_training, args=(safe_filename,))
            thread.daemon = True
            thread.start()
            
            # Update training status
            update_training_status(safe_filename, "started", step=0)
            
            logger.info(f"Training thread started for {safe_filename}")
            
            return templates.TemplateResponse(
                "retrain_started.html",
                {
                    "request": request, 
                    "filename": safe_filename,
                    "rows": rows,
                    "columns": cols
                }
            )
                
        except Exception as e:
            # Clean up the file if validation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error in file processing: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing CSV: {str(e)}")
            
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": str(e)}
        )
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": f"An unexpected error occurred: {str(e)}"}
        )

@app.get("/retrain", response_class=HTMLResponse)
async def retrain_form(request: Request):
    """Render model retraining form"""
    return templates.TemplateResponse("retrain.html", {"request": request})

# Visualization routes
@app.get("/visualize", response_class=HTMLResponse)
async def visualize(request: Request):
    """Visualization dashboard"""
    return templates.TemplateResponse("visualize.html", {"request": request})

@app.get("/api/visualization-data")
async def get_visualization_data():
    """API endpoint to get visualization data"""
    try:
        logger.info("Fetching visualization data")
        
        # Simplified data structure for testing
        data = {
            "economic_indicators": {
                "labels": ["2018", "2019", "2020", "2021", "2022", "2023"],
                "datasets": [
                    {
                        "label": "GDP Growth",
                        "data": [2.1, 2.3, -3.5, 4.2, 3.1, 2.8],
                        "borderColor": "#0d6efd",
                        "backgroundColor": "rgba(13, 110, 253, 0.1)"
                    },
                    {
                        "label": "Inflation Rate",
                        "data": [4.5, 4.8, 5.2, 6.8, 8.1, 7.2],
                        "borderColor": "#dc3545",
                        "backgroundColor": "rgba(220, 53, 69, 0.1)"
                    },
                    {
                        "label": "Exchange Rate",
                        "data": [45.2, 47.1, 52.3, 54.8, 53.2, 51.9],
                        "borderColor": "#ffc107",
                        "backgroundColor": "rgba(255, 193, 7, 0.1)"
                    }
                ]
            },
            "crisis_frequency": {
                "labels": ["East Africa", "West Africa", "North Africa", "Southern Africa", "Central Africa"],
                "datasets": [{
                    "label": "Crisis Frequency",
                    "data": [3, 5, 2, 4, 3],
                    "backgroundColor": [
                        "#0d6efd",
                        "#dc3545",
                        "#198754",
                        "#ffc107",
                        "#0dcaf0"
                    ]
                }]
            },
            "risk_factors": {
                "labels": [
                    "Exchange Rate Volatility",
                    "High Inflation",
                    "GDP Decline",
                    "External Debt",
                    "Banking Sector Weakness"
                ],
                "datasets": [{
                    "label": "Risk Impact",
                    "data": [0.78, 0.85, 0.71, 0.68, 0.91],
                    "backgroundColor": "rgba(13, 110, 253, 0.2)",
                    "borderColor": "#0d6efd"
                }]
            }
        }
        
        logger.info("Visualization data prepared successfully")
        return data
        
    except Exception as e:
        logger.error(f"Error getting visualization data: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to fetch visualization data",
                "detail": str(e),
                "traceback": traceback.format_exc()
            }
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "prediction_service": prediction_service is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

