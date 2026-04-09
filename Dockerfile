FROM python:3.11-slim

WORKDIR /app

# Install uv (the fastest way to handle your dependencies)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy your dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies without installing the project itself yet
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy the rest of your code (including the 'server' folder)
COPY . .

# HuggingFace requires port 7860
# This matches your server.app:app path perfectly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]