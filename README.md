# Banking Crisis Prediction System

A machine learning system for predicting banking crises in African countries using economic indicators.

## Live Demo

🌐 **Live Application**: [Crisis Prediction System](https://crisis-prediction.onrender.com/)

> **Note**: The application is hosted on Render's free tier, which spins down after periods of inactivity. Initial loading may take up to 50 seconds while the service spins up.

📺 **Demo Video**: [Watch on YouTube](https://youtu.be/7lD6mklIBx0)

## Project Overview

This project implements a machine learning system that:
1. Predicts the likelihood of banking crises based on economic indicators
2. Allows uploading new data for model retraining
3. Provides visualizations of crisis patterns and risk factors

## Features

1. **Prediction**
   - Make predictions on individual data points
   - Input economic indicators through a web interface
   - Get instant crisis probability predictions

2. **Data Upload & Retraining**
   - Upload new CSV datasets for model retraining
   - Automatic data preprocessing
   - Model retraining with new data
   - Uses existing model as pre-trained base

3. **Visualization**
   - Economic indicators over time
   - Regional crisis frequency analysis
   - Risk factor impact visualization

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Crisis-prediction
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the Database**
   ```bash
   python src/preprocessing.py
   ```

5. **Run the Application**
   ```bash
   python app/main.py
   ```

The application will be available at `http://localhost:8000`

## Project Structure

```
Crisis-prediction/
├── app/
│   ├── main.py           # FastAPI application
│   ├── templates/        # HTML templates
│   ├── static/          # Static files (CSS, JS)
│   └── routers/         # API routes
├── src/
│   ├── prediction.py    # Prediction service
│   ├── pipeline.py      # ML pipeline
│   ├── model.py         # Model training
│   └── preprocessing.py # Data preprocessing
├── models/              # Saved model files
├── data/               # Data directory
│   ├── uploaded/       # Uploaded files
│   ├── train/         # Training data
│   └── test/          # Test data
└── requirements.txt    # Project dependencies
```

## Usage

### Making Predictions

1. Navigate to the "Predict" page
2. Enter economic indicators:
   - Country
   - Year
   - Exchange Rate (USD)
   - GDP Weighted Default
   - Inflation Rate (CPI)
   - Systemic Crisis Status
3. Click "Make Prediction" to get results

### Uploading Data & Retraining

1. Navigate to the "Upload" page
2. Select a CSV file with new training data
3. Click "Upload"
4. After successful upload, click "Retrain Model"
5. The system will:
   - Preprocess the new data
   - Use the existing model as a pre-trained base
   - Retrain with the new data
   - Save the updated model

### Viewing Visualizations

1. Navigate to the "Visualize" page
2. Explore three types of visualizations:
   - Economic Indicators Trend
   - Regional Crisis Frequency
   - Risk Factors Impact

## Data Format

The system expects CSV files with the following columns:
- country: Country name
- year: Year of observation
- exch_usd: Exchange rate to USD
- gdp_weighted_default: GDP-weighted default rate
- inflation_annual_cpi: Annual inflation rate (CPI)
- systemic_crisis: Binary indicator (0/1)
- banking_crisis: Target variable (crisis/no_crisis)

## Model Details

- Algorithm: Random Forest Classifier
- Features: 8 selected features
- Preprocessing: 
  - Categorical encoding
  - Numerical scaling
  - Feature selection
- Evaluation metrics:
  - ROC-AUC score
  - Classification report

## License

[Your License]
