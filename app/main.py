from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks
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

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, 'static')

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
templates = Jinja2Templates(directory="templates")

# Initialize prediction service
try:
    prediction_service = PredictionService(
        model_path= os.path.join(BASE_DIR,'..','models','banking_crisis_model.pkl'),
        scaler_path=os.path.join(BASE_DIR,'..','models','scaler.pkl'),
        selected_features_path=os.path.join(BASE_DIR,'..','models','selected_features.pkl')
    )
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

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction routes
@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    """Render prediction form"""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    country: int = Form(...),
    year: int = Form(...),
    exch_usd: float = Form(...),
    gdp_weighted_default: float = Form(...),
    inflation_annual_cpi: float = Form(...),
    systemic_crisis: int = Form(0)
):
    """Make prediction based on form input"""
    try:
        # Create input data
        data = {
            "country": country,
            "year": year,
            "exch_usd": exch_usd,
            "gdp_weighted_default": gdp_weighted_default,
            "inflation_annual_cpi": inflation_annual_cpi,
            "systemic_crisis": systemic_crisis
        }
        
        # Make prediction
        if prediction_service is None:
            return templates.TemplateResponse(
                "error.html", 
                {"request": request, "message": "Prediction service not available"}
            )
        
        result = prediction_service.predict(data)
        
        # Render result page
        return templates.TemplateResponse(
            "prediction_result.html", 
            {"request": request, "result": result, "input_data": data}
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
    """Upload new data file"""
    try:
        # Create upload directory if it doesn't exist
        upload_dir = "data/uploaded"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate uploaded file
        try:
            df = pd.read_csv(file_path)
            rows, cols = df.shape
            logger.info(f"File uploaded: {file.filename}, {rows} rows, {cols} columns")
        except Exception as e:
            logger.error(f"Error reading uploaded file: {e}")
            return templates.TemplateResponse(
                "error.html", 
                {"request": request, "message": f"Invalid file format: {e}"}
            )
        
        # Return success page with file info
        return templates.TemplateResponse(
            "upload_success.html", 
            {"request": request, "filename": file.filename, "rows": rows, "cols": cols}
        )
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": str(e)}
        )

def run_training(file_path):
    """Background task to run model training"""
    try:
        logger.info(f"Starting model retraining with data: {file_path}")
        
        if ml_pipeline is None:
            logger.error("ML pipeline not available")
            return
        
        # Run training pipeline
        result = ml_pipeline.run_training_pipeline(file_path)
        
        logger.info(f"Model retraining completed: {result}")
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")

@app.post("/retrain")
async def retrain_model(request: Request, background_tasks: BackgroundTasks, filename: str = Form(...)):
    """Trigger model retraining"""
    try:
        # Get file path
        file_path = os.path.join("data/uploaded", filename)
        
        if not os.path.exists(file_path):
            return templates.TemplateResponse(
                "error.html", 
                {"request": request, "message": f"File not found: {filename}"}
            )
        
        # Start training in background
        background_tasks.add_task(run_training, file_path)
        
        # Return response
        return templates.TemplateResponse(
            "retrain_started.html", 
            {"request": request, "filename": filename}
        )
    except Exception as e:
        logger.error(f"Error starting retraining: {e}")
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "message": str(e)}
        )

# Visualization routes
@app.get("/visualize", response_class=HTMLResponse)
async def visualize(request: Request):
    """Visualization dashboard"""
    return templates.TemplateResponse("visualize.html", {"request": request})

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "prediction_service": prediction_service is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

