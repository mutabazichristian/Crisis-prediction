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
COPY data/dataset.csv /app/data/
RUN ls -la /app/data/

# Copy the rest of the application
COPY . .
COPY templates/ /app/templates/

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Train the model in the container
RUN python -c "from src.retrain import main; main()"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

