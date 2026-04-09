FROM python:3.11

WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# HuggingFace requires port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]