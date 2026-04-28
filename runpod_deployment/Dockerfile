FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install basic requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
# We assume this context comes from your Track2Cllg folder
COPY .. /app/

# Set Python path so imports work
ENV PYTHONPATH="/app"

# Command to run the Runpod serverless handler
CMD ["python", "runpod_deployment/handler.py"]
