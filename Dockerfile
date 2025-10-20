# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
