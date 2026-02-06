FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && apt-get clean

# Set working directory
WORKDIR /app

# Copy entire project
COPY . /app

# Install Python dependencies
RUN pip install --no-cache -r requirements.txt

# Install your project as a package
RUN pip install --no-cache-dir .

# # Default command (overridden by Kubeflow Trainer)
# CMD ["python" , "-m", "src.model_pipeline._07_training"]