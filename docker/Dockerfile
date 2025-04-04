FROM python:3.9-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p app/data/uploads models data

# Copy the data file first
COPY data/dataset.csv data/

# Copy the rest of the application
COPY . .
COPY templates/ /app/templates/

# Ensure proper permissions and train the model
RUN chmod +x src/retrain.py && \
    python -m src.retrain

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

