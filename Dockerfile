# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies if needed (uv is used for faster installs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration files first to leverage Docker layer caching
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
# We install openenv-core and the other requirements defined in your project
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# This includes the server/, env/, and grade/ directories
COPY . .

# Hugging Face Spaces expect the application to run on port 7860
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server using uvicorn
# We point to the 'app' object inside 'server/app.py'
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]