FROM python:3.11-slim

WORKDIR /app

# 1. Install build-system requirements first
# This prevents the 'setuptools.build-meta' missing error
RUN pip install --no-cache-dir setuptools wheel

# 2. Copy the pyproject.toml first to leverage Docker cache
COPY pyproject.toml .

# 3. Install the dependencies and the project
# This satisfies the "multi-mode deployment" requirement for the validator
RUN pip install --no-cache-dir .

# 4. Copy the application code files
COPY models.py .
COPY patients.py .
COPY environment.py .
COPY app.py .

# 5. Expose the port required by Hugging Face
EXPOSE 7860

# 6. Standard Healthcheck to ensure the server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# 7. Run the application
CMD ["python", "app.py"]