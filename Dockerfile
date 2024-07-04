# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app

COPY app.py /app/
COPY models/best_model /app/models/best_model
COPY models/feature_selector /app/models/feature_selector
COPY models/transformers /app/models/transformers
COPY templates/ /app/templates/
COPY src/logger.py /app/src/
COPY requirements_docker.txt /app/

RUN pip install --no-cache-dir -r requirements_docker.txt

EXPOSE 8000

# Set the entry point to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
